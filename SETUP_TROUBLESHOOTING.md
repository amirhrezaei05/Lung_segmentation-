# 🔧 COMPLETE SETUP & TROUBLESHOOTING GUIDE

## ⚠️ CRITICAL: Data Setup Instructions

### Problem You're Having

You're getting `FileNotFoundError` when trying to train because the data folders are empty or incorrectly structured.

---

## ✅ STEP-BY-STEP FIX

### Step 1: Understand the Required Structure

The project expects this EXACT structure:

```
lung_segmentation_project/
└── data/
    └── raw/
        ├── CXR_png/          <- Must contain .png X-ray images
        │   ├── image1.png
        │   ├── image2.png
        │   └── ...
        └── masks/            <- Must contain .png mask images
            ├── image1.png    (or image1_mask.png)
            ├── image2.png
            └── ...
```

**IMPORTANT**: 
- Image and mask filenames MUST match (e.g., `CHNCXR_0001_0.png` in both folders)
- All files must be `.png` format
- Folders must be named exactly `CXR_png` and `masks`

---

### Step 2: Download the Dataset

**From Kaggle:**
1. Go to: https://www.kaggle.com/datasets/nikhilpandey360/chest-xray-masks-and-labels
2. Click "Download" button (requires free Kaggle account)
3. Extract the downloaded ZIP file
4. You'll get a folder called `Lung Segmentation` containing:
   - `CXR_png/` - folder with X-ray images
   - `masks/` - folder with segmentation masks

---

### Step 3: Copy Data to Project

**Windows Command Prompt Method:**

```cmd
:: Navigate to your project
cd C:\Users\Amir Hossein\Music\lung_claude\lung_segmentation_project\lung_segmentation_project

:: Create data/raw folder if it doesn't exist
mkdir data\raw

:: Copy the CXR_png folder from your downloads
:: Replace "C:\Users\Amir Hossein\Downloads\Lung Segmentation" with actual path
xcopy "C:\Users\Amir Hossein\Downloads\Lung Segmentation\CXR_png" "data\raw\CXR_png" /E /I

:: Copy the masks folder
xcopy "C:\Users\Amir Hossein\Downloads\Lung Segmentation\masks" "data\raw\masks" /E /I
```

**File Explorer Method (Easier):**

1. Open File Explorer
2. Navigate to your Downloads folder where you extracted the dataset
3. Open the `Lung Segmentation` folder
4. Copy the `CXR_png` folder
5. Navigate to: `C:\Users\Amir Hossein\Music\lung_claude\lung_segmentation_project\lung_segmentation_project\data\raw\`
6. Paste `CXR_png` here
7. Go back and copy the `masks` folder
8. Paste it in the same `data\raw\` location

---

### Step 4: Verify Your Setup

Run this command to check if files are in the right place:

```cmd
:: Check CXR_png folder
dir data\raw\CXR_png

:: Check masks folder
dir data\raw\masks
```

**Expected Output:**
You should see a list of `.png` files (around 800 files in each folder).

**If you see "0 File(s)" or "File Not Found":**
- Your folders are empty or in the wrong location
- Go back to Step 3

---

### Step 5: Test the Data Loading

Create a test script to verify data loading works:

**Create `test_data.py` in project root:**

```python
from pathlib import Path
from src.data.dataset import prepare_data_paths

# Test data loading
try:
    data_dir = Path("data/raw")
    print(f"Checking data in: {data_dir.absolute()}")
    print("="*60)
    
    image_paths, mask_paths = prepare_data_paths(data_dir)
    
    print(f"\n✅ SUCCESS!")
    print(f"Found {len(image_paths)} matched pairs")
    print(f"\nFirst 3 images:")
    for i, (img, mask) in enumerate(zip(image_paths[:3], mask_paths[:3])):
        print(f"  {i+1}. Image: {img.name}")
        print(f"     Mask:  {mask.name}")
    
except Exception as e:
    print(f"\n❌ ERROR: {e}")
    print("\nPlease follow the setup guide!")
```

Run it:
```cmd
python test_data.py
```

---

## 🐛 COMMON ERRORS & FIXES

### Error 1: "Image directory not found: data\raw\CXR_png"

**Problem:** CXR_png folder doesn't exist

**Fix:**
```cmd
mkdir data\raw\CXR_png
:: Then copy your image files into this folder
```

---

### Error 2: "No PNG images found in data\raw\CXR_png"

**Problem:** CXR_png folder exists but is empty

**Fix:** 
- Download the dataset from Kaggle
- Copy the image files into `data\raw\CXR_png\`
- Make sure files are `.png` format

---

### Error 3: "No matching image-mask pairs found"

**Problem:** Image and mask filenames don't match

**Example of the issue:**
```
CXR_png/image001.png
masks/mask001.png      <- Won't match!
```

**What should work:**
```
CXR_png/image001.png
masks/image001.png     <- Matches!
```

**Fix:** 
- Make sure you're using the official dataset from Kaggle
- Don't rename files
- Both folders should have identically named files

---

### Error 4: "FileNotFoundError: [WinError 3] The system cannot find the path specified"

**Problem:** Wrong data_dir path in command

**Fix:** Check your command:
```cmd
:: ❌ WRONG - if CXR_png is directly in data/raw/
python -m src.train --config configs\efficient_unet_config.yaml --data_dir "data\raw\Lung Segmentation"

:: ✅ CORRECT - CXR_png should be at data/raw/CXR_png
python -m src.train --config configs\efficient_unet_config.yaml --data_dir data\raw
```

---

## 🎯 FINAL CHECKLIST

Before running training, verify:

- [ ] Dataset downloaded from Kaggle
- [ ] CXR_png folder exists at: `data\raw\CXR_png\`
- [ ] masks folder exists at: `data\raw\masks\`
- [ ] CXR_png folder contains ~800 .png files
- [ ] masks folder contains ~800 .png files
- [ ] Filenames match between folders
- [ ] Virtual environment activated: `venv\Scripts\activate`
- [ ] Required packages installed: `pip install -r requirements.txt`
- [ ] MLflow disabled in config (set `use_mlflow: false`)

---

## 🚀 TRAINING COMMAND

Once everything is verified:

```cmd
:: Activate virtual environment
venv\Scripts\activate

:: Train the model
python -m src.train --config configs\efficient_unet_config.yaml --data_dir data\raw
```

**Note:** Use `data\raw` NOT `data\raw\Lung Segmentation` because your structure is:
```
data\raw\CXR_png\    <- Images here
data\raw\masks\      <- Masks here
```

---

## 📊 Expected Output When Working

```
============================================================
LOADING DATA
============================================================

✓ Found 800 images in CXR_png/
✓ Found 800 masks in masks/

Matching images with masks...
✓ Successfully matched 800 image-mask pairs
  Images: C:\...\data\raw\CXR_png
  Masks: C:\...\data\raw\masks

Dataset initialized with 800 samples

Data splits:
  Train: 560 (70.0%)
  Val:   120 (15.0%)
  Test:  120 (15.0%)

============================================================
INITIALIZING MODEL
============================================================
...
```

If you see this, everything is working! 🎉

---

## 🆘 Still Having Issues?

### Quick Debug Commands

```cmd
:: 1. Check if folders exist
dir data\raw

:: 2. Check if files exist in CXR_png
dir data\raw\CXR_png | find /c ".png"

:: 3. Check if files exist in masks  
dir data\raw\masks | find /c ".png"

:: 4. Show first 5 filenames from each
dir /b data\raw\CXR_png | more
dir /b data\raw\masks | more
```

### If Nothing Works

1. **Start Fresh:**
   ```cmd
   :: Delete and recreate folders
   rmdir /s data\raw
   mkdir data\raw\CXR_png
   mkdir data\raw\masks
   ```

2. **Download dataset again** from Kaggle

3. **Copy files** using File Explorer (easier than command line)

4. **Test** with `test_data.py` script above

5. **If still failing**, send me:
   - Output of: `dir data\raw\CXR_png | find /c ".png"`
   - Output of: `dir data\raw\masks | find /c ".png"`
   - First 5 filenames from each folder

---

## 💡 Pro Tips

1. **Use absolute paths** if relative paths don't work:
   ```cmd
   python -m src.train --config configs\efficient_unet_config.yaml --data_dir "C:\Users\Amir Hossein\Music\lung_claude\lung_segmentation_project\lung_segmentation_project\data\raw"
   ```

2. **Don't use spaces in paths** - Windows can be tricky with spaces

3. **Check file extensions** - Must be `.png` not `.PNG` or `.jpg`

4. **Test small first** - Move just 10 images to test before copying all 800

---

**Follow this guide step by step and you WILL get it working!** 🚀
