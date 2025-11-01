from ase.io.extxyz import write_extxyz as _original_write_extxyz
import ase.io.extxyz
import functools

def patch_write_xyz(func):
    """装饰器来禁止write_xyz函数中第952行的fileobj.write(f'{comm}\n')执行"""
    @functools.wraps(func)
    def wrapper(fileobj, images, comment='', columns=None,
                write_info=True, write_results=True, plain=False, vec_cell=False):
        # 保存原始的fileobj.write方法
        original_write = fileobj.write
        
        # 创建一个新的write方法，跳过包含换行符的comment行
        def patched_write(text):
            # 检查是否是comment行（包含换行符且可能是comment格式）
            if text.endswith('\n') and len(text.strip()) > 0 and not text.strip().isdigit():
                # 跳过comment行的写入
                return
            # 对于其他行，使用原始方法
            return original_write(text)
        
        # 替换fileobj的write方法
        fileobj.write = patched_write
        
        try:
            # 调用原始函数
            result = func(fileobj, images, comment, columns, write_info, write_results, plain, vec_cell)
        finally:
            # 恢复原始的write方法
            fileobj.write = original_write
        
        return result
    
    return wrapper

# 应用猴子补丁
ase.io.extxyz.write_xyz = patch_write_xyz(ase.io.extxyz.write_xyz)
write_extxyz = ase.io.extxyz.write_extxyz