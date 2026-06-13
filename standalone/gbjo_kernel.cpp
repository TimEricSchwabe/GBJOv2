// Fused C++ implementation of the GBJO optimization loop.
//
// Same algorithm as gbjo_fast.py / src/optimization/methods.py::GBJO:
//   row-softmax over a (2n-2, n-1) logit matrix -> dense adjacency ->
//   CostGNNv3 forward (frozen weights) -> structural penalties + trace(expm)
//   acyclicity -> hand-derived backward w.r.t. the adjacency only ->
//   grad clip -> SGD+momentum with precomputed schedules -> per-step beam
//   projection -> deduplicated discrete scoring at the end.
//
// Differences vs the Python reference (documented, verified statistically):
//   * gemm/reduction orders differ -> last-ulp drift (chaotic loop)
//   * beam tie-breaking is deterministic ascending-index instead of Python
//     set order; the all-tied step-0 projection is passed in from Python
//
// Build (macOS, Accelerate/AMX):
//   clang++ -O3 -std=c++17 -dynamiclib -framework Accelerate \
//       -DACCELERATE_NEW_LAPACK -o libgbjo.dylib gbjo_kernel.cpp
// Build (Linux, OpenBLAS):
//   g++ -O3 -std=c++17 -shared -fPIC -o libgbjo.so gbjo_kernel.cpp -lopenblas
//   (run with OPENBLAS_NUM_THREADS=1 — threading tiny gemms hurts)

#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
typedef __LAPACK_int lapack_int_t;
#else
#include <cblas.h>
extern "C" void dgesv_(const int* n, const int* nrhs, double* a, const int* lda,
                       int* ipiv, double* b, const int* ldb, int* info);
typedef int lapack_int_t;
#endif

#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

constexpr int H = 128;          // hidden dim (CostGNNv3)
constexpr int FC1_OUT = 64;     // H/2

// Portable vectorizable expf (Cephes polynomial, ~1 ulp). Pure arithmetic +
// bit ops, so clang/gcc auto-vectorize it on NEON/AVX without libmvec or
// -ffast-math; identical numerics on macOS and Linux.
inline void vexpf_buf(float* y, const float* x, int m) {
    for (int i = 0; i < m; ++i) {
        float v = x[i];
        v = v < -87.0f ? -87.0f : (v > 88.0f ? 88.0f : v);
        const float nf = nearbyintf(v * 1.44269504088896341f);  // round(x/ln2)
        float r = v - nf * 0.693359375f;                        // ln2_hi
        r -= nf * -2.12194440e-4f;                              // ln2_lo
        float p = 1.9875691500e-4f;
        p = p * r + 1.3981999507e-3f;
        p = p * r + 8.3334519073e-3f;
        p = p * r + 4.1665795894e-2f;
        p = p * r + 1.6666665459e-1f;
        p = p * r + 5.0000001201e-1f;
        float e = p * r * r + r + 1.0f;
        int32_t bits;
        std::memcpy(&bits, &e, 4);
        bits += (int32_t)nf << 23;                              // * 2^n
        std::memcpy(&e, &bits, 4);
        y[i] = e;
    }
}

struct Ctx {
    int n_layers;
    bool dual = false;          // CostGNNDual: layer input [A^T h + h ; S h + h]
    std::vector<float> eps;     // per layer (all 0 when dual)
    std::vector<std::vector<float>> W1t, b1, W2t, b2;  // W*t: (in,out) row-major
    std::vector<float> fc1t;    // (H, FC1_OUT)
    std::vector<float> b_fc1;   // (FC1_OUT)
    std::vector<float> fc2;     // (FC1_OUT)
    float b_fc2;
};

// ---------------------------------------------------------------- gelu (erf)
// erf via Abramowitz-Stegun 7.1.26 (|abs err| < 1.5e-7 ~ float32 ulp);
// the exp is vectorized through vForce, the rest auto-vectorizes.
inline void gelu_forward(const float* x, float* erf_buf, float* y, int m) {
    constexpr float inv_sqrt2 = 0.7071067811865475f;
    constexpr float P = 0.3275911f;
    constexpr float A1 = 0.254829592f, A2 = -0.284496736f, A3 = 1.421413741f,
                    A4 = -1.453152027f, A5 = 1.061405429f;
    for (int i = 0; i < m; ++i) erf_buf[i] = -0.5f * x[i] * x[i];  // -(x/sqrt2)^2
    vexpf_buf(erf_buf, erf_buf, m);
    for (int i = 0; i < m; ++i) {
        const float za = std::fabs(x[i]) * inv_sqrt2;
        const float t = 1.0f / (1.0f + P * za);
        const float poly = ((((A5 * t + A4) * t + A3) * t + A2) * t + A1) * t;
        const float e = std::copysign(1.0f - poly * erf_buf[i], x[i]);
        erf_buf[i] = e;
        y[i] = 0.5f * x[i] * (1.0f + e);
    }
}

// dx = dy * gelu'(x), erf_buf holds erf(x/sqrt2) from the forward pass
inline void gelu_backward(const float* x, const float* erf_buf,
                          const float* dy, float* dx, float* tmp, int m) {
    constexpr float inv_sqrt2pi = 0.3989422804014327f;
    for (int i = 0; i < m; ++i) tmp[i] = -0.5f * x[i] * x[i];
    vexpf_buf(tmp, tmp, m);                               // exp(-x^2/2)
    for (int i = 0; i < m; ++i)
        dx[i] = dy[i] * (0.5f * (1.0f + erf_buf[i]) + x[i] * tmp[i] * inv_sqrt2pi);
}

// ------------------------------------------------------- expm (double, Pade13)
// dest = a*b  (all kxk row-major double)
inline void dmm(const double* a, const double* b, double* dest, int k) {
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, k, k, k,
                1.0, a, k, b, k, 0.0, dest, k);
}

void expm(const double* A, double* E, int k) {
    static const double b[] = {
        64764752532480000.0, 32382376266240000.0, 7771770303897600.0,
        1187353796428800.0, 129060195264000.0, 10559470521600.0,
        670442572800.0, 33522128640.0, 1323241920.0, 40840800.0,
        960960.0, 16380.0, 182.0, 1.0};
    const double theta13 = 5.371920351148152;

    const int kk = k * k;
    std::vector<double> As(kk);
    double norm1 = 0.0;
    for (int c = 0; c < k; ++c) {
        double s = 0.0;
        for (int r = 0; r < k; ++r) s += std::fabs(A[r * k + c]);
        norm1 = std::max(norm1, s);
    }
    int s_pow = 0;
    if (norm1 > theta13)
        s_pow = (int)std::ceil(std::log2(norm1 / theta13));
    double scale = std::ldexp(1.0, -s_pow);
    for (int i = 0; i < kk; ++i) As[i] = A[i] * scale;

    std::vector<double> A2(kk), A4(kk), A6(kk), T(kk), U(kk), V(kk), P(kk), Q(kk);
    dmm(As.data(), As.data(), A2.data(), k);
    dmm(A2.data(), A2.data(), A4.data(), k);
    dmm(A2.data(), A4.data(), A6.data(), k);

    // U = As @ (A6@(b13 A6 + b11 A4 + b9 A2) + b7 A6 + b5 A4 + b3 A2 + b1 I)
    for (int i = 0; i < kk; ++i)
        T[i] = b[13] * A6[i] + b[11] * A4[i] + b[9] * A2[i];
    dmm(A6.data(), T.data(), U.data(), k);
    for (int i = 0; i < kk; ++i)
        U[i] += b[7] * A6[i] + b[5] * A4[i] + b[3] * A2[i];
    for (int d = 0; d < k; ++d) U[d * k + d] += b[1];
    dmm(As.data(), U.data(), T.data(), k);
    U.swap(T);
    // V = A6@(b12 A6 + b10 A4 + b8 A2) + b6 A6 + b4 A4 + b2 A2 + b0 I
    for (int i = 0; i < kk; ++i)
        T[i] = b[12] * A6[i] + b[10] * A4[i] + b[8] * A2[i];
    dmm(A6.data(), T.data(), V.data(), k);
    for (int i = 0; i < kk; ++i)
        V[i] += b[6] * A6[i] + b[4] * A4[i] + b[2] * A2[i];
    for (int d = 0; d < k; ++d) V[d * k + d] += b[0];

    for (int i = 0; i < kk; ++i) { P[i] = V[i] + U[i]; Q[i] = V[i] - U[i]; }

    // solve Q X = P  (transpose to column-major, dgesv, transpose back)
    std::vector<double> Qc(kk), Pc(kk);
    for (int r = 0; r < k; ++r)
        for (int c = 0; c < k; ++c) {
            Qc[c * k + r] = Q[r * k + c];
            Pc[c * k + r] = P[r * k + c];
        }
    std::vector<lapack_int_t> ipiv(k);
    lapack_int_t kk_i = k, nrhs = k, info = 0;
    dgesv_(&kk_i, &nrhs, Qc.data(), &kk_i, ipiv.data(), Pc.data(), &kk_i, &info);
    for (int r = 0; r < k; ++r)
        for (int c = 0; c < k; ++c) E[r * k + c] = Pc[c * k + r];

    for (int p = 0; p < s_pow; ++p) {
        dmm(E, E, T.data(), k);
        std::memcpy(E, T.data(), kk * sizeof(double));
    }
}

// ----------------------------------------------------------- GNN fwd/backward
// forward; if stash != nullptr also keeps H_i (inputs) and per-layer x/erf for
// the backward pass. Returns the scalar log-cost.
struct Stash {
    std::vector<float> Hs;     // (n_layers+1, N*H) layer inputs
    std::vector<float> Us;     // (n_layers, N*H) pre-gelu activations
    std::vector<float> Es;     // (n_layers, N*H) erf(U/sqrt2)
    std::vector<float> g;      // (H) pooled
    std::vector<float> u1;     // (FC1_OUT) pre-gelu head
    std::vector<float> e1;     // (FC1_OUT) erf of head
};

float gnn_forward(const Ctx& ctx, const float* h0, const float* A,
                  const float* S, int N, Stash* st,
                  float* work /* >= (dual ? 5 : 4)*N*H */) {
    const int m = N * H;
    const int IN = ctx.dual ? 2 * H : H;  // layer-MLP input width
    float* h = work;                 // current features
    float* z = work + m;             // aggregate (N, IN)
    float* u = z + (size_t)N * IN;   // pre-activation
    float* g = u + m;                // gelu output

    std::memcpy(h, h0, m * sizeof(float));
    if (st) std::memcpy(&st->Hs[0], h, m * sizeof(float));

    for (int l = 0; l < ctx.n_layers; ++l) {
        if (ctx.dual) {
            // z = [A^T h + h ; S h + h]   (eps = 0)
            cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, N, H, N,
                        1.0f, A, N, h, H, 0.0f, z, IN);
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, N, H, N,
                        1.0f, S, N, h, H, 0.0f, z + H, IN);
            for (int r = 0; r < N; ++r)
                for (int c = 0; c < H; ++c) {
                    z[(size_t)r * IN + c] += h[r * H + c];
                    z[(size_t)r * IN + H + c] += h[r * H + c];
                }
        } else {
            const float ep = 1.0f + ctx.eps[l];
            // z = A^T h + (1+eps) h
            cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, N, H, N,
                        1.0f, A, N, h, H, 0.0f, z, H);
            for (int i = 0; i < m; ++i) z[i] += ep * h[i];
        }
        // u = z @ W1t + b1
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, N, H, IN,
                    1.0f, z, IN, ctx.W1t[l].data(), H, 0.0f, u, H);
        for (int r = 0; r < N; ++r)
            for (int c = 0; c < H; ++c) u[r * H + c] += ctx.b1[l][c];
        // g = gelu(u)
        float* erf_buf = st ? &st->Es[l * m] : g;  // reuse g as scratch if no stash
        gelu_forward(u, erf_buf, g, m);
        if (st) std::memcpy(&st->Us[l * m], u, m * sizeof(float));
        // h += g @ W2t + b2
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, N, H, H,
                    1.0f, g, H, ctx.W2t[l].data(), H, 1.0f, h, H);
        for (int r = 0; r < N; ++r)
            for (int c = 0; c < H; ++c) h[r * H + c] += ctx.b2[l][c];
        if (st) std::memcpy(&st->Hs[(l + 1) * m], h, m * sizeof(float));
    }

    // head: pool -> fc1 -> gelu -> fc2
    float gpool[H];
    for (int c = 0; c < H; ++c) {
        float s = 0.0f;
        for (int r = 0; r < N; ++r) s += h[r * H + c];
        gpool[c] = s;
    }
    float u1[FC1_OUT], a1[FC1_OUT], e1[FC1_OUT];
    cblas_sgemv(CblasRowMajor, CblasTrans, H, FC1_OUT, 1.0f, ctx.fc1t.data(),
                FC1_OUT, gpool, 1, 0.0f, u1, 1);
    for (int i = 0; i < FC1_OUT; ++i) u1[i] += ctx.b_fc1[i];
    gelu_forward(u1, e1, a1, FC1_OUT);
    float cost = ctx.b_fc2;
    for (int i = 0; i < FC1_OUT; ++i) cost += ctx.fc2[i] * a1[i];

    if (st) {
        std::memcpy(st->g.data(), gpool, sizeof(gpool));
        std::memcpy(st->u1.data(), u1, sizeof(u1));
        std::memcpy(st->e1.data(), e1, sizeof(e1));
    }
    return cost;
}

// backward of the cost w.r.t. A (dA, accumulated with coefficient 1).
// Needs the stash of the corresponding forward.
void gnn_backward(const Ctx& ctx, const Stash& st, const float* A,
                  const float* S, int N,
                  float* dA /* N*N, accumulated into */,
                  float* work /* >= (dual ? 6 : 5)*N*H */) {
    const int m = N * H;
    const int IN = ctx.dual ? 2 * H : H;
    float* dh = work;
    float* dgl = work + m;       // d gelu-out
    float* du = work + 2 * m;
    float* dz = work + 3 * m;    // (N, IN)
    float* tmp = dz + (size_t)N * IN;

    // head backward
    float du1[FC1_OUT], tmp1[FC1_OUT], dgpool[H];
    gelu_backward(st.u1.data(), st.e1.data(), ctx.fc2.data(), du1, tmp1, FC1_OUT);
    cblas_sgemv(CblasRowMajor, CblasNoTrans, H, FC1_OUT, 1.0f, ctx.fc1t.data(),
                FC1_OUT, du1, 1, 0.0f, dgpool, 1);
    for (int r = 0; r < N; ++r)
        std::memcpy(dh + r * H, dgpool, H * sizeof(float));

    for (int l = ctx.n_layers - 1; l >= 0; --l) {
        const float* Hl = &st.Hs[l * m];
        // dgl = dh @ W2t^T  (residual path keeps dh as-is)
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, N, H, H,
                    1.0f, dh, H, ctx.W2t[l].data(), H, 0.0f, dgl, H);
        // du = dgl * gelu'(U)
        gelu_backward(&st.Us[l * m], &st.Es[l * m], dgl, du, tmp, m);
        // dz = du @ W1t^T
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, N, IN, H,
                    1.0f, du, H, ctx.W1t[l].data(), H, 0.0f, dz, IN);
        if (ctx.dual) {
            // dzP = dz[:, :H], dzS = dz[:, H:]  (strided views, ld = IN)
            // dA += H_l dzP^T
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, N, N, H,
                        1.0f, Hl, H, dz, IN, 1.0f, dA, N);
            // dh += A dzP + S dzS + dzP + dzS   (S symmetric)
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, N, H, N,
                        1.0f, A, N, dz, IN, 1.0f, dh, H);
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, N, H, N,
                        1.0f, S, N, dz + H, IN, 1.0f, dh, H);
            for (int r = 0; r < N; ++r)
                for (int c = 0; c < H; ++c)
                    dh[r * H + c] += dz[(size_t)r * IN + c] +
                                     dz[(size_t)r * IN + H + c];
        } else {
            const float ep = 1.0f + ctx.eps[l];
            // dA += H_l dz^T
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, N, N, H,
                        1.0f, Hl, H, dz, H, 1.0f, dA, N);
            // dh = dh + A dz + (1+eps) dz
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, N, H, N,
                        1.0f, A, N, dz, H, 1.0f, dh, H);
            for (int i = 0; i < m; ++i) dh[i] += ep * dz[i];
        }
    }
}

// true iff the triples in `mask` form a connected subgraph via the binary
// variable-sharing adjacency S (S[t*N+u] > 0 iff triples t, u share a var).
bool subset_connected(const float* S, int N, uint32_t mask, int n) {
    if (mask == 0) return true;
    const int first = __builtin_ctz(mask);
    uint32_t seen = 1u << first, frontier = 1u << first;
    while (frontier) {
        const int u = __builtin_ctz(frontier);
        frontier &= ~(1u << u);
        for (int v = 0; v < n; ++v)
            if ((mask >> v & 1u) && !(seen >> v & 1u) &&
                S[(size_t)u * N + v] > 0.0f) {
                seen |= 1u << v;
                frontier |= 1u << v;
            }
    }
    return seen == mask;
}

// ------------------------------------------------------------------ beam
// Deterministic beam projection (ascending-index tie-break). W double (N*N).
// mask_cart: forbid attaching a table that would create a cartesian join,
// i.e. restrict to tables whose removal keeps the remaining set connected
// (so every join along the chain AND the base pair shares a variable). Falls
// back to tables sharing with the remainder, then to all, so unavoidable
// cartesians (disconnected query graphs) still produce a plan. S is the
// binary share adjacency (null when mask_cart == 0).
void beam_project(const double* W, const float* S, int mask_cart,
                  int N, int n, int beam_width, int8_t* out /* N*N */) {
    const int n_js = n - 2;             // non-root joins: n .. 2n-3
    const int root = N - 1;
    const int max_beam = 64;
    if (beam_width > max_beam) beam_width = max_beam;

    struct St { double score; int cur; uint32_t js, tb; int parent, j, t; };
    // per level states
    std::vector<std::vector<St>> levels;
    levels.reserve(n_js + 1);
    levels.push_back({{0.0, root, (n_js > 0) ? ((1u << n_js) - 1u) : 0u,
                       (uint32_t)((1ull << n) - 1ull), -1, -1, -1}});

    for (int lvl = 0; lvl < n_js; ++lvl) {
        const auto& cur = levels[lvl];
        std::vector<St> top;
        top.reserve(beam_width + 1);
        for (int si = 0; si < (int)cur.size(); ++si) {
            const St& s = cur[si];
            // cartesian-avoiding table mask: prefer tables whose removal keeps
            // the remainder connected; else tables sharing with the remainder;
            // else all (unavoidable cartesian).
            uint32_t allowed = s.tb;
            if (mask_cart) {
                uint32_t keep = 0, rem_share = 0;
                for (int t = 0; t < n; ++t) {
                    if (!(s.tb >> t & 1u)) continue;
                    const uint32_t rem = s.tb & ~(1u << t);
                    bool shares = false;
                    for (int u = 0; u < n && !shares; ++u)
                        if ((rem >> u & 1u) && S[(size_t)t * N + u] > 0.0f)
                            shares = true;
                    if (shares) rem_share |= 1u << t;
                    if (subset_connected(S, N, rem, n)) keep |= 1u << t;
                }
                if (keep) allowed = keep;
                else if (rem_share) allowed = rem_share;
            }
            for (int jb = 0; jb < n_js; ++jb) {
                if (!(s.js >> jb & 1u)) continue;
                const int j = n + jb;
                const double wj = W[j * N + s.cur];
                for (int t = 0; t < n; ++t) {
                    if (!(allowed >> t & 1u)) continue;
                    const double sc = s.score + (W[t * N + s.cur] + wj);
                    // stable insert: strictly better score ranks first; equal
                    // scores keep enumeration order
                    int pos = (int)top.size();
                    while (pos > 0 && sc > top[pos - 1].score) --pos;
                    if (pos < beam_width) {
                        St ns{sc, j, s.js & ~(1u << jb), s.tb & ~(1u << t),
                              si, j, t};
                        top.insert(top.begin() + pos, ns);
                        if ((int)top.size() > beam_width) top.pop_back();
                    }
                }
            }
        }
        levels.push_back(std::move(top));
    }

    // best final state = levels.back()[0]; backtrack edges
    std::memset(out, 0, (size_t)N * N);
    const auto& last = levels.back();
    St best = last[0];
    // remaining two tables into best.cur (j0): top-2 by W[t][j0], stable asc t
    {
        int t1 = -1, t2 = -1;
        double v1 = -1e300, v2 = -1e300;
        for (int t = 0; t < n; ++t) {
            if (!(best.tb >> t & 1u)) continue;
            const double v = W[t * N + best.cur];
            if (v > v1) { v2 = v1; t2 = t1; v1 = v; t1 = t; }
            else if (v > v2) { v2 = v; t2 = t; }
        }
        out[t1 * N + best.cur] = 1;
        out[t2 * N + best.cur] = 1;
    }
    int idx = 0;  // best is index 0 at the last level
    for (int lvl = (int)levels.size() - 1; lvl >= 1; --lvl) {
        const St& s = levels[lvl][idx];
        const St& par = levels[lvl - 1][s.parent];
        out[s.j * N + par.cur] = 1;   // join chain edge
        out[s.t * N + par.cur] = 1;   // table edge
        idx = s.parent;
    }
}

// ----------------------------------------------------- cartesian-join count
// Exact #cartesian joins of a hard left-deep plan, using the binary share
// adjacency S (S[t*N+u] > 0 iff triples t, u share a variable).
int count_carts(const int8_t* plan, const float* S, int n) {
    const int N = 2 * n - 1;
    std::vector<std::vector<int>> tch(N);
    std::vector<int> parent(N, -1);
    int base = -1;
    for (int j = n; j < N; ++j) {
        for (int r = 0; r < N; ++r)
            if (plan[(size_t)r * N + j]) {
                if (r < n) tch[j].push_back(r);
                else parent[r] = j;
            }
        if (base < 0 && tch[j].size() == 2) base = j;
    }
    std::vector<int> order(tch[base].begin(), tch[base].end());
    for (int cur = base; parent[cur] != -1; ) {
        cur = parent[cur];
        for (int t : tch[cur]) order.push_back(t);
    }
    std::vector<char> in(n, 0);
    in[order[0]] = 1;
    int carts = 0;
    for (size_t k = 1; k < order.size(); ++k) {
        bool shares = false;
        for (int u = 0; u < n && !shares; ++u)
            if (in[u] && S[(size_t)order[k] * N + u] > 0) shares = true;
        if (!shares) ++carts;
        in[order[k]] = 1;
    }
    return carts;
}

}  // namespace

// ------------------------------------------------------------------ C API
extern "C" {

void* gbjo_create(int n_layers, const float* eps,
                  const float* W1, const float* B1,   // (n_layers, H, H)/(n_layers, H); W (out,in)
                  const float* W2, const float* B2,
                  const float* fc1, const float* b_fc1,  // (FC1_OUT, H), (FC1_OUT)
                  const float* fc2, float b_fc2) {       // (FC1_OUT)
    auto* ctx = new Ctx();
    ctx->n_layers = n_layers;
    ctx->eps.assign(eps, eps + n_layers);
    for (int l = 0; l < n_layers; ++l) {
        std::vector<float> w1t(H * H), w2t(H * H);
        for (int o = 0; o < H; ++o)
            for (int i = 0; i < H; ++i) {
                w1t[i * H + o] = W1[(size_t)l * H * H + o * H + i];
                w2t[i * H + o] = W2[(size_t)l * H * H + o * H + i];
            }
        ctx->W1t.push_back(std::move(w1t));
        ctx->W2t.push_back(std::move(w2t));
        ctx->b1.emplace_back(B1 + (size_t)l * H, B1 + (size_t)(l + 1) * H);
        ctx->b2.emplace_back(B2 + (size_t)l * H, B2 + (size_t)(l + 1) * H);
    }
    ctx->fc1t.resize(H * FC1_OUT);
    for (int o = 0; o < FC1_OUT; ++o)
        for (int i = 0; i < H; ++i) ctx->fc1t[i * FC1_OUT + o] = fc1[o * H + i];
    ctx->b_fc1.assign(b_fc1, b_fc1 + FC1_OUT);
    ctx->fc2.assign(fc2, fc2 + FC1_OUT);
    ctx->b_fc2 = b_fc2;
    return ctx;
}

// CostGNNDual variant: per layer the MLP input is [A^T h + h ; S h + h]
// (eps = 0, W1 is (H, 2H)); the per-query share adjacency S is passed to
// gbjo_optimize.
void* gbjo_create_dual(int n_layers,
                       const float* W1, const float* B1,  // (L, H, 2H)/(L, H)
                       const float* W2, const float* B2,  // (L, H, H)/(L, H)
                       const float* fc1, const float* b_fc1,
                       const float* fc2, float b_fc2) {
    auto* ctx = new Ctx();
    ctx->dual = true;
    ctx->n_layers = n_layers;
    ctx->eps.assign(n_layers, 0.0f);
    const int IN = 2 * H;
    for (int l = 0; l < n_layers; ++l) {
        std::vector<float> w1t((size_t)IN * H), w2t(H * H);
        for (int o = 0; o < H; ++o) {
            for (int i = 0; i < IN; ++i)
                w1t[(size_t)i * H + o] = W1[(size_t)l * H * IN + (size_t)o * IN + i];
            for (int i = 0; i < H; ++i)
                w2t[i * H + o] = W2[(size_t)l * H * H + o * H + i];
        }
        ctx->W1t.push_back(std::move(w1t));
        ctx->W2t.push_back(std::move(w2t));
        ctx->b1.emplace_back(B1 + (size_t)l * H, B1 + (size_t)(l + 1) * H);
        ctx->b2.emplace_back(B2 + (size_t)l * H, B2 + (size_t)(l + 1) * H);
    }
    ctx->fc1t.resize(H * FC1_OUT);
    for (int o = 0; o < FC1_OUT; ++o)
        for (int i = 0; i < H; ++i) ctx->fc1t[i * FC1_OUT + o] = fc1[o * H + i];
    ctx->b_fc1.assign(b_fc1, b_fc1 + FC1_OUT);
    ctx->fc2.assign(fc2, fc2 + FC1_OUT);
    ctx->b_fc2 = b_fc2;
    return ctx;
}

void gbjo_destroy(void* p) { delete (Ctx*)p; }

// Runs the full optimization. Returns best log-cost; writes the best discrete
// adjacency into out_adj (N*N int8).
// S: (N, N) binary share adjacency (dual models only; may be null otherwise)
// lambdas: [triple_in, triple_out, join_in, join_out, acyclic, left_linear, entropy]
// lex: 0 = pick the cheapest candidate; 1 = fewest cartesian joins first,
//      then cheapest (needs S)
double gbjo_optimize(void* p, const float* h0, const float* S, int n, int steps,
                     const double* lrs, const double* moms,
                     const double* taus, const double* lts,
                     const double* lambdas, double clip, int beam_width, int lex,
                     int mask_cart,  // 1 = forbid cartesian joins in beam decode
                     const int8_t* step0_adj,  // precomputed step-0 projection (may be null)
                     int8_t* out_adj,
                     // optional candidate-pool export (Python-side selection):
                     // when out_plans != null, write up to max_pool unique
                     // candidate adjacencies + their log-costs, set *out_ncand.
                     int max_pool, int8_t* out_plans, double* out_costs,
                     int* out_ncand) {
    Ctx& ctx = *(Ctx*)p;
    const int N = 2 * n - 1;
    const int rows = 2 * n - 2, cols = n - 1;  // logit block
    const int m = N * H;

    const double lam_ti = lambdas[0], lam_to = lambdas[1], lam_ji = lambdas[2],
                 lam_jo = lambdas[3], lam_ac = lambdas[4], lam_ll = lambdas[5],
                 lam_ent = lambdas[6];

    std::vector<float> L(rows * cols, 0.0f), buf(rows * cols, 0.0f),
        W(rows * cols), A((size_t)N * N), dA((size_t)N * N), dW(rows * cols),
        work((ctx.dual ? 6 : 5) * (size_t)m);
    std::vector<double> Ad((size_t)N * N), Ed((size_t)N * N);
    Stash st;
    st.Hs.resize((ctx.n_layers + 1) * (size_t)m);
    st.Us.resize(ctx.n_layers * (size_t)m);
    st.Es.resize(ctx.n_layers * (size_t)m);
    st.g.resize(H);
    st.u1.resize(FC1_OUT);
    st.e1.resize(FC1_OUT);

    // discrete candidate pool, first-seen order
    std::unordered_map<std::string, int> seen;
    std::vector<std::vector<int8_t>> plans;

    for (int step = 0; step < steps; ++step) {
        const float inv_tau = (float)(1.0 / taus[step]);
        const double lt = lts[step];

        // --- row softmax of L*inv_tau (join rows: self col masked)
        for (int r = 0; r < rows; ++r) {
            float* wr = &W[r * cols];
            const float* lr = &L[r * cols];
            const int self_c = (r >= n) ? (r - n) : -1;
            float mx = -1e30f;
            for (int c = 0; c < cols; ++c)
                if (c != self_c) mx = std::max(mx, lr[c] * inv_tau);
            float sum = 0.0f;
            for (int c = 0; c < cols; ++c) {
                if (c == self_c) { wr[c] = 0.0f; continue; }
                wr[c] = std::exp(lr[c] * inv_tau - mx);
                sum += wr[c];
            }
            const float inv = 1.0f / sum;
            for (int c = 0; c < cols; ++c)
                if (c != self_c) wr[c] *= inv;
        }

        // --- dense adjacency
        std::fill(A.begin(), A.end(), 0.0f);
        for (int r = 0; r < rows; ++r)
            std::memcpy(&A[(size_t)r * N + n], &W[r * cols], cols * sizeof(float));

        // --- GNN forward (with stash) + backward into dA
        gnn_forward(ctx, h0, A.data(), S, N, &st, work.data());
        std::fill(dA.begin(), dA.end(), 0.0f);
        gnn_backward(ctx, st, A.data(), S, N, dA.data(), work.data());

        // --- penalties (gradients only on the parameterized block, scaled lt)
        // degree sums
        std::vector<double> in_deg(N, 0.0), out_deg(N, 0.0);
        for (int r = 0; r < N; ++r)
            for (int c = 0; c < N; ++c) {
                in_deg[c] += A[(size_t)r * N + c];
                out_deg[r] += A[(size_t)r * N + c];
            }
        // child counts per join column
        std::vector<double> ct(cols, 0.0), cj(cols, 0.0);
        for (int c = 0; c < cols; ++c) {
            for (int r = 0; r < n; ++r) ct[c] += A[(size_t)r * N + n + c];
            for (int r = n; r < N; ++r) cj[c] += A[(size_t)r * N + n + c];
        }
        // acyclicity: dA += lt*lam_ac * expm(A)^T
        for (size_t i = 0; i < (size_t)N * N; ++i) Ad[i] = A[i];
        expm(Ad.data(), Ed.data(), N);
        const float cac = (float)(lt * lam_ac);
        for (int r = 0; r < N; ++r)
            for (int c = 0; c < N; ++c)
                dA[(size_t)r * N + c] += cac * (float)Ed[(size_t)c * N + r];

        for (int r = 0; r < rows; ++r) {
            const double g_to_jo = (r < n)
                ? lam_to * 2.0 * (out_deg[r] - 1.0)        // triple rows
                : lam_jo * 2.0 * (out_deg[r] - 1.0);       // non-root join rows
            for (int c = 0; c < cols; ++c) {
                double g = g_to_jo;
                g += lam_ji * 2.0 * (in_deg[n + c] - 2.0);
                if (r < n)  // triple-child count
                    g += lam_ll * 2.0 * ((c == 0) ? (ct[0] - 2.0) : (ct[c] - 1.0));
                else        // join-child count
                    g += lam_ll * 2.0 * ((c == 0) ? cj[0] : (cj[c] - 1.0));
                if (lam_ent != 0.0) {
                    const double w = std::max((double)W[r * cols + c], 1e-10);
                    if ((double)W[r * cols + c] >= 1e-10)
                        g += lam_ent * (-(std::log(w) + 1.0));
                }
                dA[(size_t)r * N + n + c] += (float)(lt * g);
            }
        }

        // --- softmax backward into logits grad (block only)
        for (int r = 0; r < rows; ++r) {
            const float* wr = &W[r * cols];
            const float* dac = &dA[(size_t)r * N + n];
            float dot = 0.0f;
            for (int c = 0; c < cols; ++c) dot += dac[c] * wr[c];
            for (int c = 0; c < cols; ++c)
                dW[r * cols + c] = (dac[c] - dot) * wr[c] * inv_tau;
        }

        // --- clip + SGD momentum
        double nrm2 = 0.0;
        for (int i = 0; i < rows * cols; ++i) nrm2 += (double)dW[i] * dW[i];
        const double nrm = std::sqrt(nrm2);
        if (clip > 0) {
            const double coef = clip / (nrm + 1e-6);
            if (coef < 1.0)
                for (int i = 0; i < rows * cols; ++i) dW[i] *= (float)coef;
        }
        const float mo = (float)moms[step], lr = (float)lrs[step];
        for (int i = 0; i < rows * cols; ++i) {
            buf[i] = mo * buf[i] + dW[i];
            L[i] -= lr * buf[i];
        }

        // --- discrete projection + dedup
        std::vector<int8_t> plan((size_t)N * N);
        if (step == 0 && step0_adj && !mask_cart) {
            std::memcpy(plan.data(), step0_adj, (size_t)N * N);
        } else {
            for (size_t i = 0; i < (size_t)N * N; ++i) Ad[i] = A[i];
            beam_project(Ad.data(), S, mask_cart, N, n, beam_width, plan.data());
        }
        std::string key((const char*)plan.data(), (size_t)N * N);
        if (seen.emplace(key, (int)plans.size()).second)
            plans.push_back(std::move(plan));
    }

    // --- score unique discrete plans; first-seen + strict '<'
    double best_cost = 1e300;
    int best_i = 0, best_carts = 1 << 30;
    const int npool = (int)plans.size();
    for (int i = 0; i < npool; ++i) {
        std::fill(A.begin(), A.end(), 0.0f);
        for (size_t e = 0; e < (size_t)N * N; ++e) A[e] = (float)plans[i][e];
        const double c = gnn_forward(ctx, h0, A.data(), S, N, nullptr, work.data());
        if (out_plans && i < max_pool) {  // export the candidate + its log-cost
            std::memcpy(out_plans + (size_t)i * N * N, plans[i].data(),
                        (size_t)N * N);
            out_costs[i] = c;
        }
        if (lex) {
            const int cc = count_carts(plans[i].data(), S, n);
            if (cc < best_carts || (cc == best_carts && c < best_cost)) {
                best_carts = cc; best_cost = c; best_i = i;
            }
        } else if (c < best_cost) {
            best_cost = c; best_i = i;
        }
    }
    if (out_ncand) *out_ncand = (npool < max_pool) ? npool : max_pool;
    std::memcpy(out_adj, plans[best_i].data(), (size_t)N * N);
    return best_cost;
}

}  // extern "C"
