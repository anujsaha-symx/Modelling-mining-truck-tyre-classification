Set-Location -LiteralPath "D:\Tyre_Classification"
$LogFile = "D:\Tyre_Classification\outputs\classification\efficientnet\train_run.log"
"Starting training at $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" | Out-File $LogFile
python src/training/train_classifier.py --model efficientnet --epochs 25 --batch-size 32 --learning-rate 1e-3 --fine-tune --fine-tune-epoch 5 --fine-tune-learning-rate 1e-4 --early-stopping-patience 5 --scheduler-patience 2 --seed 42 2>&1 | Out-File $LogFile -Append
"Training finished at $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') with exit code $LASTEXITCODE" | Out-File $LogFile -Append
