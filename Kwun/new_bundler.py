#!/usr/bin/python3
import argparse
import os
import re
import logging
from contextlib import contextmanager

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Define context manager for changing directories
@contextmanager
def change_dir(destination):
    original_dir = os.getcwd()
    try:
        if destination:
            os.chdir(destination)
        yield
    finally:
        os.chdir(original_dir)

# Define defaults
DEFAULT_INPUT_FILEPATH = "main.cpp"
DEFAULT_INCLUDE_GUARD_PREFIX = "__INCLUDE_GUARD_"

# Launch the argument parser for a proper CLI experience
parser = argparse.ArgumentParser(
    description="A script for bundling small C++ projects into a single file by resolving includes, include guards, and accompanying .cpp files."
)
parser.add_argument(
    "-i",
    "--input",
    help="Path to the input file (typically the file containing main()). Defaults to 'main.cpp' if omitted.",
)
parser.add_argument(
    "-o",
    "--output",
    help="Path to the output bundled file. If omitted, the bundled code is printed to standard output.",
)
guardArgGroup = parser.add_mutually_exclusive_group()
guardArgGroup.add_argument(
    "-g",
    "--guard-prefix",
    help="Prefix used to detect include-guard macros. Defaults to '__INCLUDE_GUARD_' if omitted. Use an empty string ('') to treat all macros as include guards.",
)
guardArgGroup.add_argument(
    "-ng",
    "--no-guard",
    help="Do not detect include guards. Only process #include \"\" and #pragma once directives.",
    action="store_true",
)
parser.add_argument(
    "-a1",
    "--always-once",
    help="Automatically include each file only once, even without include guards or #pragma once.",
    action="store_true",
)
parser.add_argument(
    "-ns",
    "--no-source",
    help="Do not include accompanying .cpp or .c source files for included headers.",
    action="store_true",
)

arguments = parser.parse_args()
inputFilePath = arguments.input or DEFAULT_INPUT_FILEPATH
outputFilePath = arguments.output
if arguments.no_guard:
    includeGuardPrefix = None
else:
    includeGuardPrefix = arguments.guard_prefix if arguments.guard_prefix is not None else DEFAULT_INCLUDE_GUARD_PREFIX

alwaysOnce = arguments.always_once
noSource = arguments.no_source

# Set to track already included files (absolute paths)
included_files = set()

def processInclude(line, current_dir):
    """
    Process an #include directive.
    Returns the content to include or None.
    """
    include_match = re.match(r'^#include\s+"(.+)"', line)
    if include_match:
        include_file = include_match.group(1)
        include_path = os.path.abspath(os.path.join(current_dir, include_file))
        
        logging.info(f"Including file: {include_path}")
        
        if include_path in included_files and not alwaysOnce:
            logging.info(f"File already included: {include_path}")
            return ""
        
        included_files.add(include_path)
        
        if not os.path.isfile(include_path):
            logging.warning(f"Included file not found: {include_path}")
            return ""
        
        with open(include_path, 'r') as f:
            # Recursively process the included file
            bundled_content = scan(f, os.path.dirname(include_path))
        
        return bundled_content
    return None

def scan(file_obj, current_dir):
    """
    Scan a file and return its bundled content.
    """
    content = ""
    lines = file_obj.readlines()
    i = 0
    total_lines = len(lines)
    
    while i < total_lines:
        line = lines[i].rstrip('\n')
        
        # Handle #pragma once
        if re.match(r'^#pragma\s+once', line):
            abs_path = os.path.abspath(file_obj.name)
            if abs_path in included_files and not alwaysOnce:
                logging.info(f"#pragma once - already included: {abs_path}")
                return content
            included_files.add(abs_path)
            i += 1
            continue
        
        # Handle include guards
        if includeGuardPrefix:
            ifndef_match = re.match(rf'^#ifndef\s+({re.escape(includeGuardPrefix)}\w+)', line)
            if ifndef_match:
                guard_macro = ifndef_match.group(1)
                # Check next non-empty, non-comment line for #define
                j = i + 1
                while j < total_lines and re.match(r'^\s*(//.*)?$', lines[j]):
                    j += 1
                if j < total_lines:
                    define_match = re.match(rf'^#define\s+{re.escape(guard_macro)}', lines[j].rstrip('\n'))
                    if define_match:
                        # Skip #ifndef and #define lines
                        i = j + 1
                        # Now skip until corresponding #endif
                        skip_depth = 1
                        while i < total_lines and skip_depth > 0:
                            current_line = lines[i].rstrip('\n')
                            if re.match(rf'^#ifndef\s+{re.escape(guard_macro)}', current_line):
                                skip_depth += 1
                            elif re.match(r'^#endif\b', current_line):
                                skip_depth -= 1
                            i += 1
                        continue
        # Handle #include directives
        include_content = processInclude(line, current_dir)
        if include_content is not None:
            content += include_content
            i += 1
            continue
        
        # Handle other preprocessor directives that might need processing
        # For simplicity, include them as-is
        content += line + "\n"
        i += 1
    
    return content

# Main execution
def main():
    # Resolve absolute input file path
    input_abs_path = os.path.abspath(inputFilePath)
    
    if not os.path.isfile(input_abs_path):
        logging.error(f"Input file not found: {input_abs_path}")
        return
    
    included_files.add(input_abs_path)
    
    with open(input_abs_path, 'r') as f:
        bundled_code = scan(f, os.path.dirname(input_abs_path))
    
    # Output the bundled code
    if outputFilePath:
        output_abs_path = os.path.abspath(outputFilePath)
        output_dir = os.path.dirname(output_abs_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        with open(output_abs_path, 'w') as out_f:
            out_f.write(bundled_code)
        logging.info(f"Bundled code written to {outputFilePath}")
    else:
        print(bundled_code)

if __name__ == "__main__":
    main()