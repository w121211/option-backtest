# Testing

```sh
# Test all
pytest
# Test by keyword expressions
pytest tests/test_option.py -vv -k "test_find_options_expiration_gt"

# Update snapshot
pytest --snapshot-update
```
