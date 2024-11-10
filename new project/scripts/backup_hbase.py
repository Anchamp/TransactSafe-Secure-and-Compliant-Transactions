# scripts/backup_hbase.py
import os
import subprocess
import logging
from datetime import datetime
from typing import Optional, List
import shutil

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HBaseBackup:
    def __init__(self, backup_dir: str = 'backups/hbase'):
        self.backup_dir = backup_dir
        os.makedirs(backup_dir, exist_ok=True)

    def create_backup(self, tables: Optional[List[str]] = None) -> str:
        """Create a new backup of specified HBase tables"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_path = os.path.join(self.backup_dir, f'backup_{timestamp}')

        try:
            tables = tables or ['transactions', 'alerts']

            for table in tables:
                # Use HBase export utility
                export_command = [
                    'hbase',
                    'org.apache.hadoop.hbase.mapreduce.Export',
                    table,
                    os.path.join(backup_path, table)
                ]

                subprocess.run(export_command, check=True)
                logger.info(f"Successfully backed up table: {table}")

            # Compress backup
            shutil.make_archive(backup_path, 'gzip', backup_path)
            logger.info(f"Created compressed backup at: {backup_path}.tar.gz")

            return f"{backup_path}.tar.gz"

        except subprocess.CalledProcessError as e:
            logger.error(f"Error during backup: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during backup: {str(e)}")
            raise

    def restore_backup(self, backup_path: str, tables: Optional[List[str]] = None) -> None:
        """Restore HBase tables from backup"""
        try:
            # Extract backup
            extract_dir = backup_path.replace('.tar.gz', '')
            shutil.unpack_archive(backup_path, extract_dir)

            tables = tables or ['transactions', 'alerts']

            for table in tables:
                # Use HBase import utility
                import_command = [
                    'hbase',
                    'org.apache.hadoop.hbase.mapreduce.Import',
                    table,
                    os.path.join(extract_dir, table)
                ]

                subprocess.run(import_command, check=True)
                logger.info(f"Successfully restored table: {table}")

            # Cleanup
            shutil.rmtree(extract_dir)

        except subprocess.CalledProcessError as e:
            logger.error(f"Error during restore: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during restore: {str(e)}")
            raise

    def list_backups(self) -> List[str]:
        """List available backups"""
        try:
            backups = [
                f for f in os.listdir(self.backup_dir)
                if f.startswith('backup_') and f.endswith('.tar.gz')
            ]
            return sorted(backups, reverse=True)
        except Exception as e:
            logger.error(f"Error listing backups: {str(e)}")
            raise

    def cleanup_old_backups(self, keep_last: int = 5) -> None:
        """Remove old backups, keeping the specified number of most recent ones"""
        try:
            backups = self.list_backups()

            if len(backups) > keep_last:
                for backup in backups[keep_last:]:
                    backup_path = os.path.join(self.backup_dir, backup)
                    os.remove(backup_path)
                    logger.info(f"Removed old backup: {backup}")

        except Exception as e:
            logger.error(f"Error cleaning up backups: {str(e)}")
            raise


if __name__ == "__main__":
    # Example usage
    backup_manager = HBaseBackup()

    # Create new backup
    backup_path = backup_manager.create_backup()
    logger.info(f"Created backup: {backup_path}")

    # List available backups
    backups = backup_manager.list_backups()
    logger.info("Available backups:")
    for backup in backups:
        logger.info(f"  - {backup}")

    # Cleanup old backups
    backup_manager.cleanup_old_backups(keep_last=5)