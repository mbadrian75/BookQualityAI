# Side Context Entry AI - Clean Project Package v5

Project code: SCE  
Database: market_data  
Prefix: sce_

## Folder policy

Each script is inside its own section folder.  
Each section writes reports inside its own local `reports` folder.  
Shared project parameters are in `config/sce_project_config.py`.

## Run order

Run each command from project root:

```bat
cd C:\Project\BookQuality\side_context_entry_ai
python -u .\00_project_setup\00_create_sce_project_structure.py
python -u .\01_database\bootstrap\01_create_sce_meta_and_indexes.py
python -u .\01_database\inspect_raw\02_inspect_sce_raw_sources.py
```

## Report locations

- `00_project_setup/reports`
- `01_database/bootstrap/reports`
- `01_database/inspect_raw/reports`

## Important

The previous label builder package should not be used.  
Label Builder will be rebuilt inside `02_labels/entry_side/` after label parameters are locked without unapproved assumptions.
