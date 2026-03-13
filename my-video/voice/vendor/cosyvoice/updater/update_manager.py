import os
import json
import hashlib
import requests
import semver
import shutil
from pathlib import Path
import zipfile
from typing import Optional, Dict, Callable
from datetime import datetime


class UpdateManager:
    def __init__(self, current_version: str, api_endpoint: str):
        self.current_version = current_version
        self.api_endpoint = api_endpoint
        self.base_dir = Path(__file__).parent.parent
        self.temp_dir = self.base_dir / "temp"
        self.backup_dir = self.base_dir / "backup"

    def log(self, message: str, callback: Optional[Callable] = None):
        """输出日志到终端和UI"""
        print(message)
        if callback:
            callback(message)

    def check_update(self) -> Optional[Dict]:
        """检查是否有新版本可用"""
        try:
            response = requests.get(self.api_endpoint)
            response.raise_for_status()
            latest_info = response.json()

            if semver.compare(self.current_version, latest_info["version"]) < 0:
                print(f"发现新版本: {latest_info['version']}")
                print(f"更新说明: {latest_info['changelog']}")
                return latest_info
            return None

        except Exception as e:
            print(f"检查更新失败: {str(e)}")
            return None

    def verify_checksum(self, file_path: Path, expected_hash: str) -> bool:
        """验证文件完整性"""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest() == expected_hash

    def backup_current_version(
        self, progress_callback: Optional[Callable] = None
    ) -> bool:
        """备份当前版本"""
        try:
            # 获取需要更新的文件列表
            response = requests.get(self.api_endpoint)
            response.raise_for_status()
            latest_info = response.json()
            files_to_backup = latest_info.get("updated_files", [])

            if not files_to_backup:
                self.log("未指定需要更新的文件", progress_callback)
                return False

            # 创建备份目录
            self.backup_dir.mkdir(exist_ok=True)
            backup_path = self.backup_dir / f"backup_{self.current_version}"

            # 如果已存在同版本备份，先删除
            if backup_path.exists():
                shutil.rmtree(backup_path)

            self.log("开始备份当前版本...", progress_callback)

            # 备份每个文件
            for file_path in files_to_backup:
                src_file = self.base_dir / file_path
                dest_file = backup_path / file_path

                if src_file.exists():
                    # 确保目标目录存在
                    dest_file.parent.mkdir(parents=True, exist_ok=True)

                    # 复制文件或目录
                    if src_file.is_dir():
                        shutil.copytree(src_file, dest_file)
                    else:
                        shutil.copy2(src_file, dest_file)
                    self.log(f"已备份: {file_path}", progress_callback)

            # 保存版本信息
            version_info = {
                "version": self.current_version,
                "backup_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "files": files_to_backup,
            }
            with open(backup_path / "version_info.json", "w", encoding="utf-8") as f:
                json.dump(version_info, f, ensure_ascii=False, indent=2)

            self.log(f"备份完成: {backup_path}", progress_callback)
            return True

        except Exception as e:
            self.log(f"备份失败: {str(e)}", progress_callback)
            return False

    def download_update(
        self, url: str, checksum: str, progress_callback: Optional[Callable] = None
    ) -> Optional[Path]:
        """下载更新包"""
        try:
            # 创建临时目录
            self.temp_dir.mkdir(exist_ok=True)
            file_path = self.temp_dir / "update.zip"

            # 分块下载
            self.log("开始下载更新...", progress_callback)
            response = requests.get(url, stream=True)
            response.raise_for_status()

            total_size = int(response.headers.get("content-length", 0))
            with open(file_path, "wb") as f:
                downloaded = 0
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        progress = (
                            (downloaded / total_size) * 100 if total_size > 0 else 0
                        )
                        self.log(f"下载进度: {progress:.1f}%", progress_callback)

            # 验证checksum
            if self.verify_checksum(file_path, checksum):
                self.log("下载完成，文件验证通过", progress_callback)
                return file_path
            else:
                self.log("文件校验失败", progress_callback)
                return None

        except Exception as e:
            self.log(f"下载更新失败: {str(e)}", progress_callback)
            return None

    def apply_update(
        self, update_file: Path, progress_callback: Optional[Callable] = None
    ) -> bool:
        """应用更新"""
        try:
            # 获取更新信息
            response = requests.get(self.api_endpoint)
            response.raise_for_status()
            latest_info = response.json()
            files_to_update = latest_info.get("updated_files", [])

            if not files_to_update:
                self.log("更新包中未指定需要更新的文件", progress_callback)
                return False

            # 先备份需要更新的文件
            self.log("正在备份当前版本...", progress_callback)
            if not self.backup_current_version(progress_callback):
                return False

            # 解压更新包
            self.log("正在解压更新包...", progress_callback)
            extracted_dir = self.temp_dir / "extracted"
            with zipfile.ZipFile(update_file, "r") as zip_ref:
                zip_ref.extractall(extracted_dir)

            # 更新指定的文件
            self.log("正在更新文件...", progress_callback)
            for file_path in files_to_update:
                src_file = extracted_dir / file_path
                dest_file = self.base_dir / file_path

                if src_file.exists():
                    # 确保目标目录存在
                    dest_file.parent.mkdir(parents=True, exist_ok=True)

                    if dest_file.exists():
                        # 如果是文件就删除文件,如果是目录就保留
                        if not dest_file.is_dir():
                            dest_file.unlink()
                        else:
                            shutil.rmtree(dest_file)

                    # 复制新文件
                    if src_file.is_dir():
                        shutil.copytree(src_file, dest_file)
                    else:
                        shutil.copy2(src_file, dest_file)
                    self.log(f"已更新: {file_path}", progress_callback)
                else:
                    self.log(
                        f"警告: 更新包中不存在文件: {file_path}", progress_callback
                    )

            # 更新版本号
            version_file = self.base_dir / "version.py"
            with open(version_file, "w", encoding="utf-8") as f:
                f.write(f'VERSION = "{latest_info["version"]}"\n')
            self.log(f"版本号已更新为: {latest_info['version']}", progress_callback)

            self.log("更新完成", progress_callback)
            return True

        except Exception as e:
            self.log(f"更新失败: {str(e)}", progress_callback)
            self.rollback(progress_callback)
            return False

    def rollback(self, progress_callback: Optional[Callable] = None) -> bool:
        """回滚到上一个版本"""
        try:
            # 查找所有备份
            backups = list(self.backup_dir.glob("backup_*"))
            if not backups:
                self.log("没有找到可用的备份", progress_callback)
                return False

            # 按版本号排序备份
            def get_version(path):
                try:
                    version = path.name.split("_")[1]
                    return semver.VersionInfo.parse(version)
                except:
                    return semver.VersionInfo.parse("0.0.0")

            backups.sort(key=get_version, reverse=True)
            latest_backup = backups[0]

            # 读取备份的版本信息
            version_info_file = latest_backup / "version_info.json"
            if not version_info_file.exists():
                self.log("备份版本信息不存在", progress_callback)
                return False

            with open(version_info_file, "r", encoding="utf-8") as f:
                version_info = json.load(f)

            self.log(f"正在回滚到版本: {version_info['version']}", progress_callback)
            self.log(f"备份时间: {version_info['backup_time']}", progress_callback)

            # 恢复备份的文件
            for file_path in version_info["files"]:
                src_file = latest_backup / file_path
                dest_file = self.base_dir / file_path

                if src_file.exists():
                    # 确保目标目录存在
                    dest_file.parent.mkdir(parents=True, exist_ok=True)

                    # 恢复文件或目录
                    if src_file.is_dir():
                        if dest_file.exists():
                            shutil.rmtree(dest_file)
                        shutil.copytree(src_file, dest_file)
                    else:
                        if dest_file.exists():
                            dest_file.unlink()
                        shutil.copy2(src_file, dest_file)
                    self.log(f"已恢复: {file_path}", progress_callback)

            # 更新版本号
            version_file = self.base_dir / "version.py"
            with open(version_file, "w", encoding="utf-8") as f:
                f.write(f'VERSION = "{version_info["version"]}"\n')
            self.log(f"版本号已更新为: {version_info['version']}", progress_callback)

            self.log("回滚完成", progress_callback)
            return True

        except Exception as e:
            self.log(f"回滚失败: {str(e)}", progress_callback)
            return False

    def cleanup(self):
        """清理临时文件"""
        try:
            if self.temp_dir.exists():
                shutil.rmtree(self.temp_dir)
        except Exception as e:
            print(f"清理临时文件失败: {str(e)}")
