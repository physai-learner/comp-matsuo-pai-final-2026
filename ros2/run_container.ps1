# 講義（第4・5回）の演習環境と同じ ROS 2 Humble のコンテナを起動する。
# 違いはコンテナ名と、このリポジトリを /workspace にマウントすることだけ。
#
#   .\ros2\run_container.ps1      # 初回は作成、2回目以降は起動するだけ
#   ブラウザで http://localhost:6080/ を開く（rerun は 9090・9876）
$ErrorActionPreference = "Stop"
$name = "pai_final"
$image = "airobotbook/ros2-desktop-ai-robot-book-humble:latest"
$repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path

$exists = docker ps -a --filter "name=^$name$" --format "{{.Names}}"
if ($exists) {
    docker start $name | Out-Null
} else {
    docker run -d --name $name `
        -p 6080:80 --shm-size=512m `
        -p 9090:9090 -p 9876:9876 `
        -v "${repo}:/workspace" `
        $image | Out-Null
}
Write-Host "起動した: http://localhost:6080/"
