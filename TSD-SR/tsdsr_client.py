#!/usr/bin/env python3
"""
TSD-SR Client
Simple client to interact with TSD-SR Worker Server
"""

import requests
import argparse
import sys
from pathlib import Path
from typing import Optional
import json

class TSDSRClient:
    def __init__(self, host: str = "127.0.0.1", port: int = 8090):
        self.base_url = f"http://{host}:{port}"
        self.session = requests.Session()
    
    def health_check(self) -> bool:
        """Check if server is running"""
        try:
            response = self.session.get(f"{self.base_url}/health")
            return response.status_code == 200
        except Exception as e:
            print(f"❌ Health check failed: {e}")
            return False
    
    def get_info(self) -> dict:
        """Get server information"""
        try:
            response = self.session.get(f"{self.base_url}/info")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"❌ Failed to get info: {e}")
            return {}
    
    def get_stats(self) -> dict:
        """Get server statistics"""
        try:
            response = self.session.get(f"{self.base_url}/stats")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"❌ Failed to get stats: {e}")
            return {}
    
    def upscale_image(
        self,
        image_path: str,
        output_path: Optional[str] = None,
        upscale: int = 2,
        process_size: int = 128
    ) -> bool:
        """
        Upscale a single image
        
        Args:
            image_path: Path to input image
            output_path: Path to save output (default: same as input with _upscaled suffix)
            upscale: Upscale factor (2 or 4)
            process_size: Processing size (64-256)
        """
        try:
            # Validate input
            input_file = Path(image_path)
            if not input_file.exists():
                print(f"❌ Input file not found: {image_path}")
                return False
            
            if not input_file.is_file():
                print(f"❌ Input is not a file: {image_path}")
                return False
            
            # Determine output path
            if output_path is None:
                output_file = input_file.parent / f"{input_file.stem}_upscaled{input_file.suffix}"
            else:
                output_file = Path(output_path)
            
            print(f"\n📥 Uploading image: {image_path}")
            print(f"   Size: {input_file.stat().st_size / (1024*1024):.2f}MB")
            print(f"   Upscale: {upscale}x, Process size: {process_size}")
            
            # Send request
            with open(image_path, 'rb') as f:
                files = {'file': f}
                params = {'upscale': upscale, 'process_size': process_size}
                response = self.session.post(
                    f"{self.base_url}/upscale",
                    files=files,
                    params=params,
                    timeout=300  # 5 minute timeout
                )
            
            if response.status_code == 200:
                # Save output
                with open(output_file, 'wb') as f:
                    f.write(response.content)
                
                output_size = output_file.stat().st_size / (1024*1024)
                output_dims = response.headers.get('X-Output-Size', 'Unknown')
                
                print(f"\n✅ Upscaling successful!")
                print(f"   Output file: {output_file}")
                print(f"   Size: {output_dims}")
                print(f"   File size: {output_size:.2f}MB\n")
                return True
            else:
                print(f"❌ Request failed: {response.status_code}")
                print(f"   {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ Error: {e}")
            return False
    
    def upscale_batch(
        self,
        image_dir: str,
        output_dir: Optional[str] = None,
        upscale: int = 2,
        process_size: int = 128,
        pattern: str = "*.png"
    ) -> int:
        """
        Upscale multiple images
        
        Args:
            image_dir: Directory containing images
            output_dir: Directory to save outputs (default: same as input)
            upscale: Upscale factor (2 or 4)
            process_size: Processing size (64-256)
            pattern: File pattern to match (e.g., "*.png", "*.jpg")
        
        Returns:
            Number of successfully processed images
        """
        try:
            input_path = Path(image_dir)
            if not input_path.is_dir():
                print(f"❌ Input directory not found: {image_dir}")
                return 0
            
            # Find images
            images = list(input_path.glob(pattern))
            if not images:
                print(f"❌ No images found matching {pattern} in {image_dir}")
                return 0
            
            print(f"\n📁 Found {len(images)} images to process")
            
            output_path = Path(output_dir) if output_dir else input_path
            output_path.mkdir(parents=True, exist_ok=True)
            
            successful = 0
            for idx, image_file in enumerate(images, 1):
                output_file = output_path / f"{image_file.stem}_upscaled{image_file.suffix}"
                
                print(f"\n[{idx}/{len(images)}] Processing {image_file.name}...")
                
                if self.upscale_image(str(image_file), str(output_file), upscale, process_size):
                    successful += 1
            
            print(f"\n{'='*60}")
            print(f"Batch processing complete!")
            print(f"Successfully processed: {successful}/{len(images)}")
            print(f"{'='*60}\n")
            
            return successful
            
        except Exception as e:
            print(f"❌ Error: {e}")
            return 0

def main():
    parser = argparse.ArgumentParser(
        description="TSD-SR Client - Upscale images using TSD-SR Worker Server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Upscale single image
  python tsdsr_client.py -i image.png -o upscaled.png
  
  # Upscale with 4x factor
  python tsdsr_client.py -i image.png -u 4
  
  # Check server status
  python tsdsr_client.py --info
  
  # Upscale batch of images
  python tsdsr_client.py --batch ./images --output ./upscaled
  
  # Get server stats
  python tsdsr_client.py --stats
        """
    )
    
    parser.add_argument(
        "-i", "--input",
        type=str,
        help="Input image file"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        help="Output image file"
    )
    parser.add_argument(
        "-u", "--upscale",
        type=int,
        default=2,
        choices=[2, 4],
        help="Upscale factor (default: 2)"
    )
    parser.add_argument(
        "-ps", "--process-size",
        type=int,
        default=128,
        help="Process size (default: 128)"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Server host (default: 127.0.0.1)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8090,
        help="Server port (default: 8090)"
    )
    parser.add_argument(
        "--info",
        action="store_true",
        help="Get server information"
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Get server statistics"
    )
    parser.add_argument(
        "--health",
        action="store_true",
        help="Check server health"
    )
    parser.add_argument(
        "--batch",
        type=str,
        help="Batch process directory"
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.png",
        help="File pattern for batch processing (default: *.png)"
    )
    
    args = parser.parse_args()
    
    # Create client
    client = TSDSRClient(host=args.host, port=args.port)
    
    print(f"\n{'='*60}")
    print("TSD-SR Client")
    print(f"{'='*60}\n")
    
    # Handle different commands
    if args.health:
        print("Checking server health...")
        if client.health_check():
            print("✅ Server is running")
        else:
            print("❌ Server is not responding")
        return
    
    if args.info:
        print("Fetching server information...")
        info = client.get_info()
        if info:
            print(json.dumps(info, indent=2))
        else:
            print("❌ Failed to get server information")
        return
    
    if args.stats:
        print("Fetching server statistics...")
        stats = client.get_stats()
        if stats:
            print(json.dumps(stats, indent=2))
        else:
            print("❌ Failed to get server statistics")
        return
    
    if args.batch:
        successful = client.upscale_batch(
            args.batch,
            args.output,
            args.upscale,
            args.process_size,
            args.pattern
        )
        sys.exit(0 if successful > 0 else 1)
    
    if args.input:
        success = client.upscale_image(
            args.input,
            args.output,
            args.upscale,
            args.process_size
        )
        sys.exit(0 if success else 1)
    
    # If no command specified
    parser.print_help()

if __name__ == "__main__":
    main()

