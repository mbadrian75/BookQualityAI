@echo off
REM run_sce_v2_channel_zone_pipeline.bat
REM Put this file in: C:\Project\BookQuality\side_context_entry_ai\99_run
REM Then run it from the project root or double-click after checking paths.

cd /d C:\Project\BookQuality\side_context_entry_ai

echo ============================================================
echo STEP 1 - DRY RUN cleanup
echo ============================================================
python 01_database\cleanup\03_clean_generated_collections.py --db market_data

echo.
echo If the DRY-RUN report is correct, press any key to APPLY cleanup.
pause

echo ============================================================
echo STEP 2 - APPLY cleanup
echo ============================================================
python 01_database\cleanup\03_clean_generated_collections.py --db market_data --apply

echo ============================================================
echo STEP 3 - Build Channel-Zone lower entry features
echo ============================================================
python 02_features\channel_zone\01_build_channel_zone_features_v2.py ^
  --db market_data ^
  --m5 m5 ^
  --m15 m15 ^
  --out sce_features_channel_zone_entry_m5_v2 ^
  --drop-output

echo.
echo ============================================================
echo NEXT STEPS
echo ============================================================
echo Now run the existing higher-timeframe context / dataset / label / train scripts
echo with the new lower feature collection:
echo sce_features_channel_zone_entry_m5_v2
echo.
pause
