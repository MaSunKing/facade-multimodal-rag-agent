"""Process-local upload policy. Restart after changing environment configuration."""
import os


def positive_setting(name: str, default: int, maximum: int) -> int:
    value = os.getenv(name, str(default))
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(f'{name} must be a positive integer') from exc
    if not 1 <= parsed <= maximum:
        raise ValueError(f'{name} must be between 1 and {maximum}')
    return parsed


MAX_SESSION_FILES = positive_setting('CUSTOMER_UPLOAD_MAX_FILES', 4, 32)
MAX_FILE_BYTES = positive_setting('CUSTOMER_UPLOAD_MAX_FILE_MB', 25, 512) * 1024**2
MAX_SESSION_BYTES = positive_setting('CUSTOMER_UPLOAD_MAX_TOTAL_MB', 50, 1024) * 1024**2


def public_upload_policy() -> dict:
    return {'max_files': MAX_SESSION_FILES, 'max_file_bytes': MAX_FILE_BYTES,
            'max_total_bytes': MAX_SESSION_BYTES,
            'scope': 'active_attachment_session',
            'note': '上传额度不是解析完整性或模型上下文容量保证。'}
