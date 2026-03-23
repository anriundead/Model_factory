#!/bin/bash
# 在 mergeKit_beta 所在服务器上运行，用于检查 Samba 共享与目录权限
set -e
echo "=== 1. 用户与路径 ==="
echo "当前用户: $(whoami)"
PROJECT_ROOT="/home/a/ServiceEndFiles/Workspaces/mergeKit_beta"
echo "项目目录: $PROJECT_ROOT"
if [ -f "$PROJECT_ROOT/app.db" ]; then
  ls -la "$PROJECT_ROOT/app.db"
else
  echo "app.db 不存在（应用未启动过则正常）"
fi
echo ""
echo "=== 2. 目录权限（应为 a 或当前用户可写）==="
ls -ld "$PROJECT_ROOT"
echo ""
echo "=== 3. Samba 配置中 [mergekit] 段 ==="
if grep -q '^\[mergekit\]' /etc/samba/smb.conf 2>/dev/null; then
  grep -A 20 '^\[mergekit\]' /etc/samba/smb.conf
else
  echo "未找到 [mergekit] 段，请在 /etc/samba/smb.conf 手动补充共享配置"
fi
echo ""
echo "=== 4. Samba 用户 a 是否存在 ==="
if sudo pdbedit -L 2>/dev/null | grep -qE '^a\s'; then
  echo "Samba 用户 a 已存在"
else
  echo "Samba 中无用户 a，请执行: sudo smbpasswd -a a"
fi
echo ""
echo "=== 5. smbd 状态 ==="
systemctl is-active smbd 2>/dev/null || echo "smbd 未运行或不可用"
echo ""
echo "=== 6. 从本机测试列出共享（会提示输入 a 的 Samba 密码）==="
smbclient -L //localhost/mergekit -U a 2>/dev/null || echo "需输入密码或检查 smbd 与防火墙"
