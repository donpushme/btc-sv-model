#!/usr/bin/env python3
"""
Cleanup Script: Remove old timestamped model files
Helps clean up the models directory by removing old timestamped files
and keeping only the new simple naming convention.
"""

import os
import glob
import shutil
from datetime import datetime

def cleanup_models_directory():
    """Clean up the models directory by removing old timestamped files."""
    print("🧹 Cleaning up models directory...")
    
    models_dir = "models"
    if not os.path.exists(models_dir):
        print("❌ Models directory not found")
        return
    
    # List all files in models directory
    all_files = os.listdir(models_dir)
    print(f"📁 Found {len(all_files)} files in models directory")
    
    # Files to keep (new simple naming convention)
    files_to_keep = []
    files_to_remove = []
    
    for file in all_files:
        file_path = os.path.join(models_dir, file)
        
        # Keep files with simple naming convention
        if any(file.startswith(f"{crypto}_model.") for crypto in ['BTC', 'ETH', 'XAU', 'SOL']):
            files_to_keep.append(file)
            print(f"  ✅ Keeping: {file}")
        # Keep the old best_model.pth for reference (but warn about it)
        elif file == "best_model.pth":
            files_to_keep.append(file)
            print(f"  ⚠️ Keeping (old system): {file}")
        # Remove timestamped files
        elif any(pattern in file for pattern in ['_model_', '_feature_engineer_', '_metadata_']):
            files_to_remove.append(file)
            print(f"  🗑️ Removing: {file}")
        else:
            # Keep other files (unknown)
            files_to_keep.append(file)
            print(f"  ❓ Keeping (unknown): {file}")
    
    if not files_to_remove:
        print("✅ No timestamped files to remove")
        return
    
    # Create backup directory
    backup_dir = f"models_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(backup_dir, exist_ok=True)
    
    # Move files to backup instead of deleting
    print(f"\n💾 Moving {len(files_to_remove)} files to backup: {backup_dir}")
    for file in files_to_remove:
        src = os.path.join(models_dir, file)
        dst = os.path.join(backup_dir, file)
        shutil.move(src, dst)
        print(f"  📦 Moved: {file}")
    
    print(f"\n✅ Cleanup completed!")
    print(f"📁 Files kept in models directory: {len(files_to_keep)}")
    print(f"📦 Files moved to backup: {len(files_to_remove)}")
    print(f"💾 Backup location: {backup_dir}")
    
    if "best_model.pth" in files_to_keep:
        print(f"\n⚠️ Note: 'best_model.pth' is from the old single-BTC system")
        print(f"   It's not compatible with the new multi-crypto system")
        print(f"   You can safely delete it if you have new models trained")

def show_current_models():
    """Show the current state of the models directory."""
    print("\n📊 Current Models Directory Status:")
    print("=" * 50)
    
    models_dir = "models"
    if not os.path.exists(models_dir):
        print("❌ Models directory not found")
        return
    
    all_files = os.listdir(models_dir)
    
    # Group files by crypto
    crypto_files = {}
    other_files = []
    
    for file in all_files:
        if file.startswith(('BTC_', 'ETH_', 'XAU_', 'SOL_')):
            crypto = file.split('_')[0]
            if crypto not in crypto_files:
                crypto_files[crypto] = []
            crypto_files[crypto].append(file)
        else:
            other_files.append(file)
    
    # Show crypto-specific files
    for crypto in ['BTC', 'ETH', 'XAU', 'SOL']:
        if crypto in crypto_files:
            print(f"\n{crypto} Models:")
            for file in crypto_files[crypto]:
                print(f"  📄 {file}")
        else:
            print(f"\n{crypto} Models: ❌ None found")
    
    # Show other files
    if other_files:
        print(f"\nOther Files:")
        for file in other_files:
            print(f"  📄 {file}")

def main():
    """Main cleanup function."""
    print("🧹 Model Directory Cleanup Script")
    print("=" * 40)
    
    # Show current state
    show_current_models()
    
    # Ask for confirmation
    print(f"\n❓ Do you want to clean up timestamped files? (y/n): ", end="")
    response = input().lower().strip()
    
    if response in ['y', 'yes']:
        cleanup_models_directory()
        print(f"\n📊 Final status:")
        show_current_models()
    else:
        print("❌ Cleanup cancelled")

if __name__ == "__main__":
    main() 