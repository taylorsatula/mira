import os
import shutil

def delete_pycache_directories(base_path):
	"""
	Recursively delete all __pycache__ directories in the specified base path.
	
	:param base_path: Full path to the project directory
	"""
	# Expand the full path to handle any potential home directory shortcuts
	full_path = os.path.expanduser(base_path)
	
	# Counter for deleted directories
	deleted_count = 0
	
	# Walk through the directory
	for root, dirs, files in os.walk(full_path):
		if '__pycache__' in dirs:
			pycache_path = os.path.join(root, '__pycache__')
			try:
				shutil.rmtree(pycache_path)
				print(f"Deleted: {pycache_path}")
				deleted_count += 1
			except Exception as e:
				print(f"Error deleting {pycache_path}: {e}")
	
	# Print summary
	print(f"\nTotal __pycache__ directories deleted: {deleted_count}")

# Specify the exact path
project_path = '/Users/taylut/Programming/botwithmemory'

if __name__ == '__main__':
	delete_pycache_directories(project_path)