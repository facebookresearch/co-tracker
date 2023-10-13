### Progress Bar widget 
### Original work https://github.com/tqdm/tqdm


# Usage



#### in Python
```python
from supervisely.app.widgets import SlyTqdm

progress_bar = SlyTqdm(message='Waiting for Start')

for _ in progress_bar(range(10), message='Iterations'):
    pass
```

#### in HTML
```html
<div> {{{ progress_bar.to_html() }}} </div>
```

