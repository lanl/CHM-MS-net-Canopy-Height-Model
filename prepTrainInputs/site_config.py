"""
Site Configuration Loader for SatCHM

This module provides utilities to load site-specific configuration from sites.json,
eliminating the need to manually copy .env files when switching between sites.

Usage:
    from site_config import load_site_config
    
    config = load_site_config('ws')
    epsg = config['epsg']
    api_key = config['openTopoAPIkey']
"""

import json
import os
from pathlib import Path

def load_site_config(site_code):
    """
    Load configuration for a specific site from sites.json
    
    Args:
        site_code (str): Site code (e.g., 'ws', 'qm', 'lm')
        
    Returns:
        dict: Site configuration including epsg, api keys, paths, etc.
        
    Raises:
        FileNotFoundError: If sites.json doesn't exist
        KeyError: If site_code is not found in sites.json
    """
    # Find sites.json in project root
    script_dir = Path(__file__).resolve().parent.parent
    sites_json_path = script_dir / 'sites.json'
    
    if not sites_json_path.exists():
        raise FileNotFoundError(
            f"sites.json not found at {sites_json_path}\n"
            "Please create sites.json in the project root directory."
        )
    
    # Load sites.json
    with open(sites_json_path, 'r') as f:
        sites = json.load(f)
    
    # Check if site exists
    if site_code not in sites:
        available_sites = ', '.join(sites.keys())
        raise KeyError(
            f"Site '{site_code}' not found in sites.json\n"
            f"Available sites: {available_sites}"
        )
    
    # Get site config
    config = sites[site_code].copy()
    
    # Add computed paths
    project_root = script_dir
    config['site'] = site_code
    config['inferenceShpPath'] = str(
        project_root / 'downloads' / site_code / 'infShp' / config['inference_shape']
    )
    
    return config


def get_site_env_vars(site_code):
    """
    Get environment variables dict for a site (for backward compatibility with .env)
    
    Args:
        site_code (str): Site code (e.g., 'ws', 'qm', 'lm')
        
    Returns:
        dict: Dictionary of environment variable names to values
    """
    config = load_site_config(site_code)
    
    # Map to .env variable names
    env_vars = {
        'site': config['site'],
        'epsg': str(config['epsg']),
        'openTopoAPIkey': config['openTopoAPIkey'],
        'inferenceShpPath': config['inferenceShpPath'],
    }
    
    # Optional fields
    if 'customTrainShpPath' in config:
        env_vars['customTrainShpPath'] = config['customTrainShpPath']
    if 'customLidarTifPath' in config:
        env_vars['customLidarTifPath'] = config['customLidarTifPath']
    if 'maxarAPIkey' in config:
        env_vars['maxarAPIkey'] = config['maxarAPIkey']
    
    return env_vars


def list_available_sites():
    """
    List all available sites in sites.json
    
    Returns:
        list: List of site codes
    """
    script_dir = Path(__file__).resolve().parent
    sites_json_path = script_dir / 'sites.json'
    
    if not sites_json_path.exists():
        return []
    
    with open(sites_json_path, 'r') as f:
        sites = json.load(f)
    
    return list(sites.keys())


def print_site_info(site_code):
    """
    Print information about a site
    
    Args:
        site_code (str): Site code (e.g., 'ws', 'qm', 'lm')
    """
    config = load_site_config(site_code)
    
    print(f"\n{'='*60}")
    print(f"Site: {site_code} - {config.get('full_name', 'Unknown')}")
    print(f"{'='*60}")
    print(f"EPSG: {config['epsg']}")
    print(f"Inference Shape: {config.get('inference_shape', 'Not specified')}")
    print(f"Inference Path: {config['inferenceShpPath']}")
    if 'notes' in config:
        print(f"Notes: {config['notes']}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    # Test the module
    print("Available sites:", list_available_sites())
    
    for site in list_available_sites():
        try:
            print_site_info(site)
        except Exception as e:
            print(f"Error loading site {site}: {e}")

