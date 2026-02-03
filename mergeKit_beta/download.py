import os
import shutil
import ssl
import urllib3
from modelscope.hub.snapshot_download import snapshot_download

# ================= 配置区域 =================
# 你想要下载的模型 ID
REPO_ID = 'KaLM-Embedding/KaLM-embedding-multilingual-mini-instruct-v2.5'

# 最终存放的根目录
MODELS_POOL_DIR = './models_pool'
# ===========================================

# --- 1. 网络环境自动修复 (针对你的服务器环境) ---
# 清除可能导致报错的代理设置
os.environ['http_proxy'] = ''
os.environ['https_proxy'] = ''
os.environ['HTTP_PROXY'] = ''
os.environ['HTTPS_PROXY'] = ''
# 忽略 SSL 证书验证 (解决 EOF 报错)
ssl._create_default_https_context = ssl._create_unverified_context
urllib3.disable_warnings()

def clean_download(repo_id, target_root):
    # 1. 计算最终想要的清爽路径
    # 例如: 从 'Qwen/Qwen2.5-0.5B-Instruct' 提取出 'Qwen2.5-0.5B-Instruct'
    clean_name = repo_id.split('/')[-1]
    final_path = os.path.join(target_root, clean_name)

    # 检查是否已存在
    if os.path.exists(final_path):
        print(f"⚠️  目录已存在，跳过下载: {final_path}")
        return

    print(f"🚀 开始下载: {repo_id}")
    print(f"📂 目标路径: {final_path}")

    # 2. 设置一个临时下载目录 (用来接住 ModelScope 乱七八糟的目录结构)
    temp_cache_dir = "./temp_download_cache"
    
    try:
        # 3. 执行下载
        # ModelScope 会下载到: ./temp_download_cache/Qwen/Qwen2___5-0___5B.../
        actual_download_path = snapshot_download(
            repo_id, 
            cache_dir=temp_cache_dir, 
            revision='master'
        )
        
        print(f"✅ 下载完成 (临时路径): {actual_download_path}")

        # 4. 创建最终的父目录
        if not os.path.exists(target_root):
            os.makedirs(target_root)

        # 5. 核心步骤：移动并重命名
        # 把下载好的文件夹里面的内容，直接“搬”到我们要的 final_path
        print("📦 正在整理目录结构...")
        shutil.move(actual_download_path, final_path)
        
        print(f"✨ 成功！模型已就位: {final_path}")

    except Exception as e:
        print(f"❌ 下载或整理失败: {e}")
    finally:
        # 6. 清理现场：删除那个临时的缓存目录
        if os.path.exists(temp_cache_dir):
            print("🧹 清理临时文件...")
            shutil.rmtree(temp_cache_dir)

if __name__ == '__main__':
    clean_download(REPO_ID, MODELS_POOL_DIR)
