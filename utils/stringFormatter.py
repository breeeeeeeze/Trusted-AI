def stringFormatter(string, **kwargs):
    for k, v in kwargs.items():
        try:
            string.replace(f'{{{k}}}', f'{v}')
        except Exception:
            pass
