# Project Cleanup Script
# Moves obsolete and experimental files to ARCHIVE folder

Write-Host "Starting project cleanup..." -ForegroundColor Green

# Create ARCHIVE folder if it doesn't exist
if (!(Test-Path "ARCHIVE")) {
    New-Item -ItemType Directory -Name "ARCHIVE"
    Write-Host "Created ARCHIVE folder" -ForegroundColor Yellow
}

# List of files to archive (obsolete/experimental versions)
$filesToArchive = @(
    # Original paper files
    "1842_fullpaper_1035931648545198.pdf",
    "RAW.tex",
    "IMPROVED_GROT_NET.tex",
    
    # Obsolete architecture files
    "grot_net_architecture.py",
    "memory_optimized_genius.py",
    
    # Obsolete preprocessing files
    "preprocess_stft_slices.py",
    "preprocess_stft_slices_v2.py",
    
    # Obsolete training files
    "train_grot_net.py",
    "train_genius_architecture.py",
    "train_genius_real_data.py",
    "train_memory_optimized.py",
    
    # Obsolete evaluation files
    "evaluate_grot_net.py",
    "evaluate_genius_model.py",
    "evaluate_ultra_compact.py",
    
    # Debug and test files
    "debug_dataset.py",
    "debug_model.py",
    "debug_model_analysis.png",
    "simple_debug.py",
    "simple_debug_analysis.png",
    "test_model.py",
    "test_crawler.py",
    
    # Obsolete comparison files
    "compare_approaches.py",
    "simple_test_comparison.py",
    
    # Obsolete engine simulation files
    "engine_simulation_test.py",
    "engine_simulation_test_results.json",
    "engine_simulation_comprehensive_test.png",
    
    # Obsolete STFTSC files
    "stftsc_visualization.py",
    "stftsc_real_data_performance.png",
    "stftsc_real_data_results.json",
    "stftsc_simplified_diagram.png",
    "stftsc_test_results.png",
    "test_stftsc_real_data.py",
    
    # Obsolete frequency analysis files
    "frequency_analysis_rpm_800.png",
    "frequency_analysis_rpm_1200.png",
    "frequency_analysis_rpm_2000.png",
    "frequency_analysis_rpm_3000.png",
    "frequency_based_rpm.py",
    
    # Obsolete model checkpoints (keep only the best one)
    "ultra_compact_checkpoint_epoch_10.pth",
    "ultra_compact_checkpoint_epoch_20.pth",
    "ultra_compact_checkpoint_epoch_30.pth",
    "ultra_compact_checkpoint_epoch_40.pth",
    "ultra_compact_checkpoint_epoch_50.pth",
    
    # Obsolete evaluation results
    "ultra_compact_evaluation_results.png",
    "ultra_compact_evaluation_results.json",
    "rpm_estimation_comparison.png",
    
    # Obsolete documentation files
    "CRAWLER_ENHANCEMENTS.md",
    "DATASET_STRATEGY.md",
    "EXPERIMENT_ROADMAP.md",
    "HL_CEAD_SUMMARY.md",
    "PREPROCESSING_SUCCESS.md",
    
    # Obsolete data processing files
    "create_rpm_mapping.py",
    "dataset_integration.py",
    "data_preprocessing.py",
    "preprocess_hl_cead.py",
    
    # Obsolete crawler files
    "youtube_engine_crawler.py",
    
    # Obsolete setup files
    "setup_experiment.py",
    "fix_grot_net.py"
)

# Move files to ARCHIVE
$movedCount = 0
$skippedCount = 0

foreach ($file in $filesToArchive) {
    if (Test-Path $file) {
        try {
            Move-Item $file "ARCHIVE\" -Force
            Write-Host "Moved: $file" -ForegroundColor Green
            $movedCount++
        }
        catch {
            Write-Host "Failed to move: $file" -ForegroundColor Red
        }
    } else {
        Write-Host "Skipped (not found): $file" -ForegroundColor Yellow
        $skippedCount++
    }
}

Write-Host "`nCleanup Summary:" -ForegroundColor Cyan
Write-Host "Files moved: $movedCount" -ForegroundColor Green
Write-Host "Files skipped: $skippedCount" -ForegroundColor Yellow

# List remaining important files
Write-Host "`nRemaining important files:" -ForegroundColor Cyan
$importantFiles = @(
    "paper/TRANSFORMER_RPM_ESTIMATION.tex",
    "paper/NVH.bib",
    "ultra_compact_genius.py",
    "train_ultra_compact.py",
    "preprocess_simple.py",
    "stftsc_implementation.py",
    "compare_methods.py",
    "comprehensive_engine_comparison.py",
    "simple_engine_test.py",
    "stftsc_energy_beam_visualization.py",
    "best_ultra_compact_model.pth",
    "ultra_compact_detailed_analysis.png",
    "ultra_compact_evaluation_metrics.png",
    "ultra_compact_training_curves.png",
    "comprehensive_engine_comparison.png",
    "comprehensive_engine_comparison_results.json",
    "simple_engine_test_visualization.png",
    "simple_engine_test_results.json",
    "stftsc_algorithm_flowchart.png",
    "stftsc_energy_beam_visualization.png",
    "README.md",
    "ARCHITECTURE.md",
    "config.json",
    "requirements.txt",
    "docker-compose.yml",
    "Dockerfile",
    "comparison_analysis.md"
)

foreach ($file in $importantFiles) {
    if (Test-Path $file) {
        Write-Host "✓ $file" -ForegroundColor Green
    }
}

Write-Host "`nCleanup completed!" -ForegroundColor Green 