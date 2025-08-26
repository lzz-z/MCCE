# -----------------------------------------------------------------
# problem/sacs/sacs_file_modifier.py
# -----------------------------------------------------------------
import re
import shutil
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path
import logging


# (保留原有的 AdvancedSacsFileModifier 和 SacsFileModifier 类定义，但在下面新增/修改方法)

class SacsFileModifier:
    def __init__(self, project_path: str):
        # ... (原有的 __init__ 代码保持不变) ...
        self.project_path = Path(project_path)
        self.input_file = self.project_path / "sacinp.demo06"
        self.backup_dir = self.project_path / "backups"
        self.logger = logging.getLogger(self.__class__.__name__)
        self.backup_dir.mkdir(exist_ok=True)
        if not self.input_file.exists():
            raise FileNotFoundError(f"SACS input file not found: {self.input_file}")

    def _create_backup(self) -> Optional[Path]:
        """Creates a backup of the current input file."""
        try:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = self.backup_dir / f"sacinp_pre_eval_{ts}.demo06"
            shutil.copy2(self.input_file, backup_path)
            self.logger.info(f"Created backup: {backup_path.name}")
            return backup_path
        except Exception as e:
            self.logger.error(f"Failed to create backup: {e}")
            return None

    def _restore_from_backup(self, backup_path: Path):
        """Restores the input file from a backup."""
        try:
            shutil.copy2(backup_path, self.input_file)
            self.logger.warning(f"Restored file from backup: {backup_path.name}")
        except Exception as e:
            self.logger.error(f"Failed to restore from backup {backup_path.name}: {e}")

    # --- 新增功能：提取代码块 ---
    def extract_code_blocks(self, block_prefixes: List[str]) -> Dict[str, str]:
        """
        Extracts full lines of SACS code based on a list of unique prefixes.

        Args:
            block_prefixes: A list of prefixes, e.g., ["GRUP LG1", "PGRUP P01"].
                            It's assumed each prefix uniquely identifies one line in the file.

        Returns:
            A dictionary mapping the prefix (used as an identifier) to the full code line.
        """
        code_blocks = {}
        try:
            with open(self.input_file, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()

            for prefix in block_prefixes:
                found = False
                for line in lines:
                    # We use .strip().startswith() to be robust against leading spaces
                    if line.strip().startswith(prefix):
                        # Use a simplified key for the dictionary (e.g., "GRUP_LG1")
                        key = prefix.replace(" ", "_")
                        code_blocks[key] = line.rstrip('\n')
                        found = True
                        break  # Assume first match is the correct one
                if not found:
                    self.logger.warning(f"Could not find a unique code block for prefix: '{prefix}'")

        except Exception as e:
            self.logger.error(f"Error extracting code blocks: {e}")
        return code_blocks

    # --- 新增功能：替换代码块 ---
    def replace_code_blocks(self, new_code_blocks: Dict[str, str]) -> bool:
        """
        Replaces entire lines in the SACS file with new code blocks.

        Args:
            new_code_blocks: A dictionary mapping block identifiers (e.g., "GRUP_LG1")
                             to the new, full code lines.

        Returns:
            True if successful, False otherwise.
        """
        backup_path = self._create_backup()
        if not backup_path:
            return False

        try:
            with open(self.input_file, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()

            original_lines = lines[:]
            lines_replaced = 0

            for identifier, new_line in new_code_blocks.items():
                prefix = identifier.replace("_", " ")
                line_found_and_replaced = False
                for i, line in enumerate(lines):
                    if line.strip().startswith(prefix):
                        self.logger.info(
                            f"Replacing block '{prefix}':\n  OLD: {line.strip()}\n  NEW: {new_line.strip()}")
                        lines[i] = new_line + '\n'
                        lines_replaced += 1
                        line_found_and_replaced = True
                        break  # Move to the next identifier

                if not line_found_and_replaced:
                    self.logger.warning(f"Identifier '{identifier}' from LLM not found in SACS file. Skipping.")

            if lines_replaced == 0:
                self.logger.error("LLM provided identifiers, but none matched the SACS file. No changes made.")
                self._restore_from_backup(backup_path)
                return False

            with open(self.input_file, 'w', encoding='utf-8', errors='ignore') as f:
                f.writelines(lines)

            self.logger.info(f"Successfully replaced {lines_replaced} code blocks.")
            return True

        except Exception as e:
            self.logger.critical(f"Fatal error during code block replacement: {e}")
            self._restore_from_backup(backup_path)
            return False
