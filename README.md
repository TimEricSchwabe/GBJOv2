## Getting started
### Using uv
0. [Install uv](https://docs.astral.sh/uv/getting-started/installation/#standalone-installer)

1. Create virtual environment:
    ```bash
    uv venv
    uv sync
    ```

# Some notes

## Using Data Version Control
[Docs](https://dvc.org/)

0. [Installing](https://dvc.org/doc/start)

1. Add config data to the `.dvc/config.local` file

2. Pulling data ([`dvc pull`](https://dvc.org/doc/command-reference/pull)):
    ```bash
    dvc pull
    ```

    or specific file:
    ```bash
    dvc pull filename_without_dvc_extension
    ```

3. Adding new data ([`dvc add`](https://dvc.org/doc/command-reference/add)):
    ```bash
    # Adding all data in the folder:
    dvc add --glob FOLDER_PATH/**/*.*

    # Adding specific file:
    dvc add FOLDER_PATH/FILE_NAME
    ```

4. Updating data ([`dvc commit`](https://dvc.org/doc/command-reference/commit)): 
    ```bash
    dvc commit
    ```

## Virtuoso notes
### 0. `.ini` file
Should be located at `Virtuoso_INSTALLATION_FOLDER/database/virtuoso.ini`

Some useful entries:
1. `DefaultHost = ...` - the endpoint to the Virtuoso server, openable in a browser
2. `DirsAllowed = ..., ..., ...` - directories that are allowed to be accessed by the server

### 1. Loading data into Virtuoso
Using the command line tool `isql` to load data into Virtuoso:
```sql
SQL> ld_dir_all('PATH_TO_RDF_DATA_FOLDER', '*.*', 'GRAPHNAME');
SQL> rdf_loader_run();
```

**Known issues:**
* [Access to 'PATH_TO_RDF_DATA_FOLDER' is denied due to access control in ini file](https://stackoverflow.com/questions/76451769/virtuoso-bulk-insert-failed)


### 2. Python connection
**Known issues:**

* [2s waiting time](https://stackoverflow.com/questions/69853221/querying-local-sparql-endpoint-is-very-slow)