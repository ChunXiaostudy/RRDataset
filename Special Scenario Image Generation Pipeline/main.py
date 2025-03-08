import os
import argparse

def main():
    parser = argparse.ArgumentParser(description='Generate prompts and images')
    parser.add_argument('--step', choices=['prompts', 'images', 'all'], 
                       default='all', help='Which step to run')
    args = parser.parse_args()
    
    if args.step in ['prompts', 'all']:
        print("\n=== Generating Prompts ===")
        from generate_prompts import main as generate_prompts
        generate_prompts()
    
    if args.step in ['images', 'all']:
        print("\n=== Generating Images ===")
        from generate_images import main as generate_images
        generate_images()

if __name__ == "__main__":
    main() 