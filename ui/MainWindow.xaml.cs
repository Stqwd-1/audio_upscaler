// Copyright 2026 Stanislav Suharkov
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

using System;
using System.Diagnostics;
using System.IO;
using System.Windows;
using Microsoft.Win32;

namespace AudioUpscaler;

public partial class MainWindow : Window
{
    private string _pythonExe = "python";
    private string _scriptPath = "";

    public MainWindow()
    {
        InitializeComponent();
        Loaded += MainWindow_Loaded;
        TransientStrength.ValueChanged += (_, e) =>
        {
            TransientValue.Text = e.NewValue.ToString("F1");
        };
    }

    private void MainWindow_Loaded(object sender, RoutedEventArgs e)
    {
        // Auto-detect python path relative to project
        var baseDir = AppDomain.CurrentDomain.BaseDirectory;
        var projectDir = Path.GetFullPath(Path.Combine(baseDir, "..", "..", "..", ".."));

        // Try venv python first
        var venvPython = Path.Combine(projectDir, ".venv", "Scripts", "python.exe");
        if (File.Exists(venvPython))
            _pythonExe = venvPython;

        _scriptPath = Path.Combine(projectDir, "inference.py");
        if (!File.Exists(_scriptPath))
        {
            // Try from current directory
            _scriptPath = Path.Combine(Directory.GetCurrentDirectory(), "inference.py");
        }

        StatusText.Text = $"Python: {_pythonExe}\nScript: {_scriptPath}";
    }

    private void BrowseInput_Click(object sender, RoutedEventArgs e)
    {
        var dialog = new OpenFileDialog
        {
            Filter = "Audio Files|*.wav;*.mp3;*.flac;*.ogg;*.aac;*.m4a;*.wma;*.opus;*.aiff|All Files|*.*",
            Title = "Select Input Audio"
        };
        if (dialog.ShowDialog() == true)
            InputPath.Text = dialog.FileName;
    }

    private void BrowseCheckpoint_Click(object sender, RoutedEventArgs e)
    {
        var dialog = new OpenFileDialog
        {
            Filter = "PyTorch Checkpoint|*.pt;*.pth|All Files|*.*",
            Title = "Select Model Checkpoint"
        };
        if (dialog.ShowDialog() == true)
            CheckpointPath.Text = dialog.FileName;
    }

    private void BrowseOutput_Click(object sender, RoutedEventArgs e)
    {
        var dialog = new System.Windows.Forms.FolderBrowserDialog
        {
            Description = "Select Output Directory"
        };
        if (dialog.ShowDialog() == System.Windows.Forms.DialogResult.OK)
            OutputDir.Text = dialog.SelectedPath;
    }

    private async void Upscale_Click(object sender, RoutedEventArgs e)
    {
        if (string.IsNullOrWhiteSpace(InputPath.Text) || InputPath.Text == "No file selected...")
        {
            MessageBox.Show("Please select an input file.", "Error", MessageBoxButton.OK, MessageBoxImage.Warning);
            return;
        }

        if (!File.Exists(_scriptPath))
        {
            MessageBox.Show($"inference.py not found at:\n{_scriptPath}", "Error", MessageBoxButton.OK, MessageBoxImage.Error);
            return;
        }

        // Build arguments
        var targetRates = new[] { 96000, 192000, 384000 };
        var targetRate = targetRates[TargetRate.SelectedIndex];

        var quantModes = new[] { "none", "int8", "fp16" };
        var quantMode = quantModes[QuantizeMode.SelectedIndex];

        var checkpoint = string.IsNullOrWhiteSpace(CheckpointPath.Text)
            ? "checkpoints/best_model.pt"
            : CheckpointPath.Text;
        var outputFile = Path.Combine(
            string.IsNullOrWhiteSpace(OutputDir.Text) ? "outputs" : OutputDir.Text,
            $"{Path.GetFileNameWithoutExtension(InputPath.Text)}_{targetRate}hz.wav");

        var args = $"\"{_scriptPath}\" \"{InputPath.Text}\"";
        args += $" -c \"{checkpoint}\"";
        args += $" -r {targetRate}";
        args += $" -o \"{outputFile}\"";

        if (UseQC.IsChecked == true)
        {
            args += " --qc";
            args += $" --n-candidates {(int)CandidateCount.Value}";
            args += $" --min-agreement {MinAgreement.Value:F2}";
        }

        if (UseTTA.IsChecked == true)
            args += " --tta";

        if (TransientStrength.Value > 0)
            args += $" --transient-strength {TransientStrength.Value:F1}";

        if (quantMode != "none")
            args += $" --quantize {quantMode}";

        // Disable button, show progress
        UpscaleBtn.IsEnabled = false;
        ProgressBar.IsIndeterminate = true;
        StatusText.Text = $"Running: python {args}\n";

        try
        {
            var psi = new ProcessStartInfo
            {
                FileName = _pythonExe,
                Arguments = args,
                WorkingDirectory = Path.GetDirectoryName(_scriptPath) ?? ".",
                UseShellExecute = false,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                CreateNoWindow = true,
            };

            using var process = Process.Start(psi);
            if (process == null)
            {
                StatusText.Text = "Failed to start process.";
                return;
            }

            process.OutputDataReceived += (_, ev) =>
            {
                if (ev.Data != null)
                    Dispatcher.Invoke(() => StatusText.Text += ev.Data + "\n");
            };
            process.ErrorDataReceived += (_, ev) =>
            {
                if (ev.Data != null)
                    Dispatcher.Invoke(() => StatusText.Text += ev.Data + "\n");
            };

            process.BeginOutputReadLine();
            process.BeginErrorReadLine();

            await process.WaitForExitAsync();

            StatusText.Text += process.ExitCode == 0
                ? "\nDone!"
                : $"\nFailed with exit code {process.ExitCode}";
        }
        catch (Exception ex)
        {
            StatusText.Text = $"Error: {ex.Message}";
        }
        finally
        {
            UpscaleBtn.IsEnabled = true;
            ProgressBar.IsIndeterminate = false;
        }
    }
}
