@echo off
REM STEP 03 - Validate Channel-Zone Features
REM Put this package content in: C:\Project\BookQuality\side_context_entry_ai

cd /d C:\Project\BookQuality\side_context_entry_ai

python 02_features\channel_zone\02_validate_channel_zone_features_v2.py ^
  --db market_data ^
  --features sce_features_channel_zone_entry_m5_v2

pause
