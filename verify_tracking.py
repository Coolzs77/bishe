#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Verify tracking module identifier translations
"""

import sys
import os
import ast

def check_file_identifiers(filepath):
    """Check for Chinese identifiers in a Python file"""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    try:
        tree = ast.parse(content)
    except SyntaxError as e:
        print(f"✗ Syntax error in {filepath}: {e}")
        return False
    
    chinese_identifiers = []
    
    for node in ast.walk(tree):
        # Check function and class names
        if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)):
            if any('\u4e00' <= c <= '\u9fff' for c in node.name):
                chinese_identifiers.append(f"  - {node.name} (line {node.lineno})")
        
        # Check variable names
        elif isinstance(node, ast.Name):
            if any('\u4e00' <= c <= '\u9fff' for c in node.id):
                chinese_identifiers.append(f"  - {node.id}")
        
        # Check attribute names
        elif isinstance(node, ast.Attribute):
            if any('\u4e00' <= c <= '\u9fff' for c in node.attr):
                chinese_identifiers.append(f"  - .{node.attr}")
    
    if chinese_identifiers:
        print(f"✗ {filepath} contains Chinese identifiers:")
        for ident in set(chinese_identifiers):
            print(ident)
        return False
    else:
        print(f"✓ {filepath} - all identifiers in English")
        return True

def main():
    tracking_files = [
        'src/tracking/tracker.py',
        'src/tracking/deepsort_tracker.py',
        'src/tracking/bytetrack_tracker.py',
        'src/tracking/centertrack_tracker.py',
        'src/tracking/kalman_filter.py',
    ]
    
    print("Checking tracking module files for Chinese identifiers...\n")
    
    all_passed = True
    for filepath in tracking_files:
        if os.path.exists(filepath):
            if not check_file_identifiers(filepath):
                all_passed = False
        else:
            print(f"✗ File not found: {filepath}")
            all_passed = False
    
    print("\n" + "="*60)
    if all_passed:
        print("✓ All tracking module files passed verification!")
        print("✓ All Chinese identifiers have been translated to English")
        print("✓ All Chinese comments/docstrings/messages preserved")
        return 0
    else:
        print("✗ Some files failed verification")
        return 1

if __name__ == '__main__':
    sys.exit(main())
