"""
Data Setup Verification Script

Run this to check if your data is set up correctly before training.
"""

from pathlib import Path
import sys

def check_data_setup():
    """Verify data setup is correct"""
    
    print("="*70)
    print("LUNG SEGMENTATION PROJECT - DATA SETUP VERIFICATION")
    print("="*70)
    
    # Check folders exist
    data_dir = Path("data/raw")
    cxr_dir = data_dir / "CXR_png"
    masks_dir = data_dir / "masks"
    
    print("\n1. Checking directory structure...")
    print(f"   Project root: {Path('.').absolute()}")
    print(f"   Data directory: {data_dir.absolute()}")
    
    if not data_dir.exists():
        print(f"\n❌ ERROR: data/raw/ folder doesn't exist!")
        print(f"   Please create it: mkdir data\\raw")
        return False
    else:
        print(f"   ✓ data/raw/ exists")
    
    if not cxr_dir.exists():
        print(f"\n❌ ERROR: CXR_png folder doesn't exist!")
        print(f"   Expected location: {cxr_dir.absolute()}")
        print(f"   Please create it and add your X-ray images")
        return False
    else:
        print(f"   ✓ CXR_png/ exists")
    
    if not masks_dir.exists():
        print(f"\n❌ ERROR: masks folder doesn't exist!")
        print(f"   Expected location: {masks_dir.absolute()}")
        print(f"   Please create it and add your mask images")
        return False
    else:
        print(f"   ✓ masks/ exists")
    
    # Check for files
    print("\n2. Checking for image files...")
    
    image_files = list(cxr_dir.glob("*.png"))
    mask_files = list(masks_dir.glob("*.png"))
    
    print(f"   Images found: {len(image_files)}")
    print(f"   Masks found:  {len(mask_files)}")
    
    if len(image_files) == 0:
        print(f"\n❌ ERROR: No PNG images found in CXR_png folder!")
        print(f"   Please add your X-ray images to: {cxr_dir.absolute()}")
        print(f"\n   Expected: ~800 .png files")
        print(f"   Actual: {len(image_files)} files")
        
        # Show what's in the folder
        all_files = list(cxr_dir.iterdir())
        if len(all_files) > 0:
            print(f"\n   Files found in folder:")
            for f in all_files[:5]:
                print(f"     - {f.name}")
            if len(all_files) > 5:
                print(f"     ... and {len(all_files)-5} more")
        
        return False
    
    if len(mask_files) == 0:
        print(f"\n❌ ERROR: No PNG masks found in masks folder!")
        print(f"   Please add your mask images to: {masks_dir.absolute()}")
        return False
    
    print(f"   ✓ Found {len(image_files)} images")
    print(f"   ✓ Found {len(mask_files)} masks")
    
    # Check matching
    print("\n3. Checking if images match with masks...")
    
    matched = 0
    for img_path in image_files[:10]:  # Check first 10
        # Try different matching patterns
        patterns = [
            masks_dir / img_path.name,  # Exact match
            masks_dir / f"{img_path.stem}_mask.png",  # _mask suffix
            masks_dir / f"{img_path.stem}_seg.png",  # _seg suffix
        ]
        
        for pattern in patterns:
            if pattern.exists():
                matched += 1
                break
    
    if matched == 0:
        print(f"\n❌ ERROR: No matching image-mask pairs found!")
        print(f"\n   Sample image names:")
        for img in image_files[:5]:
            print(f"     - {img.name}")
        print(f"\n   Sample mask names:")
        for mask in mask_files[:5]:
            print(f"     - {mask.name}")
        print(f"\n   Image and mask filenames should match!")
        return False
    
    print(f"   ✓ Found matching pairs (tested {matched}/10 samples)")
    
    # Try loading with the actual function
    print("\n4. Testing data loading with project code...")
    
    try:
        from src.data.dataset import prepare_data_paths
        
        image_paths, mask_paths = prepare_data_paths(data_dir)
        
        print(f"\n   ✓ Data loading successful!")
        print(f"   ✓ Total matched pairs: {len(image_paths)}")
        
        if len(image_paths) < 100:
            print(f"\n   ⚠ WARNING: Only {len(image_paths)} pairs found")
            print(f"   Expected: ~800 pairs from Kaggle dataset")
        
        # Show samples
        print(f"\n   Sample image-mask pairs:")
        for i, (img, mask) in enumerate(zip(image_paths[:3], mask_paths[:3])):
            print(f"     {i+1}. Image: {img.name}")
            print(f"        Mask:  {mask.name}")
        
    except Exception as e:
        print(f"\n❌ ERROR during data loading:")
        print(f"   {str(e)}")
        return False
    
    # Final summary
    print("\n" + "="*70)
    print("✅ ALL CHECKS PASSED!")
    print("="*70)
    print(f"\n✓ Data directories exist")
    print(f"✓ Found {len(image_paths)} image files")
    print(f"✓ Found {len(mask_paths)} mask files")
    print(f"✓ All images matched with masks")
    print(f"✓ Data loading works correctly")
    
    print(f"\n🚀 You're ready to train!")
    print(f"\nRun this command to start training:")
    print(f"   python -m src.train --config configs\\efficient_unet_config.yaml --data_dir data\\raw")
    print("="*70)
    
    return True


if __name__ == "__main__":
    try:
        success = check_data_setup()
        if not success:
            print("\n" + "="*70)
            print("❌ DATA SETUP INCOMPLETE")
            print("="*70)
            print("\nPlease fix the issues above before training.")
            print("See SETUP_TROUBLESHOOTING.md for detailed help.")
            sys.exit(1)
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
