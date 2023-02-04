from pathlib import Path
import os

path = Path(__file__).parent
print(path)
print(str(path))
print(type(path))

path1 = os.path.join(path, 'utils')
print(path1)

path2 = os.path.join(path.parent, 'annotations/trimaps')
print(path2)