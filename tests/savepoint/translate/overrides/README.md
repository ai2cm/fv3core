# Threshold overrides

`--threshold_overrides_file` takes in a yaml file with error thresholds specified for specific backend and platform configuration. Currently, two types of error overrides are allowed: maximum error and near zero.

For maximum error, a blanket `max_error` is specified to override the parent classes relative error threshold.

For near zero override, `ignore_near_zero_errors` is specified to allow some fields to pass with higher relative error if the absolute error is very small.

Override yaml file should have one of the following formats:

## One near zero value for all variables

```Stencil_name:
 - backend: <backend>
   max_error: <value>
   near_zero: <value>
   ignore_near_zero_errors:
    - <var>
    - <var2>
    - ...
```
## Variable specific near zero value

```Stencil_name:
 - backend: <backend>
   max_error: <value>
   near_zero: <value>
   ignore_near_zero_errors:
    <var>:<value>
    <var2>:<value>
    ...
```
