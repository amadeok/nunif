import os, sys , argparse

os.chdir(os.path.dirname(__file__))

import argparse

def main():
    parser = argparse.ArgumentParser(description='Process input and output files')
    
    # Add input argument
    parser.add_argument('-i', '--input', 
                       required=True,
                       help='Input file path')
    
    # Add output argument
    parser.add_argument('-o', '--output',
                       required=True,
                       help='Output file path')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Access the values
    input_file = args.input
    output_file = args.output
    
    print(f"Input file: {input_file}")
    print(f"Output file: {output_file}")
    cmd = rf"""call F:\all\GitHub\nunif\venv\Scripts\activate.bat && python -m stlizer -i "{input_file}" """
    cmd += f""" -o "{output_file}"  --iteration 200 --smoothing 4 --batch-size=8 """
    os.system(cmd)

if __name__ == '__main__':
    main()

