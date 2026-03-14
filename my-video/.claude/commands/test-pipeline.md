# Digital Human Pipeline - Standardized End-to-End Test

Run the full digital human video generation pipeline test with resource monitoring and report generation.

## Arguments
- $ARGUMENTS - Target material directory path and/or Douyin URL, separated by space. Examples:
  - `/path/to/materials`
  - `https://v.douyin.com/xxxxx`
  - `/path/to/materials https://v.douyin.com/xxxxx`

## Instructions

You are running a standardized digital human pipeline test. Parse the arguments to determine:
1. **Material directory**: Any argument that looks like a file path (starts with `/` or `~`)
2. **Douyin URL**: Any argument containing `douyin.com`

### Test Procedure

Execute the test using the Python test script at `tests/test_pipeline.py` in the project root.

**Step 1: Validate inputs**
- If a material directory is provided, verify it exists and list its contents (video files)
- If a Douyin URL is provided, confirm it's a valid URL format

**Step 2: Run the pipeline test**
Run the test script with the conda Python environment:
```bash
cd /home/claude/workspace/video-model/my-video
/home/claude/miniconda3/envs/myvideo/bin/python tests/test_pipeline.py \
  [--material-dir /path/to/dir] \
  [--douyin-url "URL"] \
  --mode step_by_step
```

Use `--material-dir` if a directory was provided, `--douyin-url` if a URL was provided. Both can be used together.

**Important**: Do NOT truncate videos. Use full-length source materials for testing.

**Step 3: Monitor the test**
- The test runs with background resource monitoring (CPU/GPU/memory)
- Let it run to completion - it may take several minutes for the digital human generation step
- Use a timeout of 600000ms (10 minutes)

**Step 4: Read and present the report**
After the test completes:
1. Find the generated markdown report in the output directory (`test_output/test_report_*.md`)
2. Read it and present the full report to the user
3. Also note the JSON report path for programmatic access

**Step 5: Summarize**
Provide a concise summary highlighting:
- Overall pass/fail status
- Time breakdown per step
- Peak CPU/GPU utilization
- Any failures or warnings
- Output file locations and sizes

### Error Handling
- If any step fails, the pipeline continues to report the failure but stops dependent steps
- Always present whatever report was generated, even if partial
- If the script itself fails to run, diagnose the error and report it
