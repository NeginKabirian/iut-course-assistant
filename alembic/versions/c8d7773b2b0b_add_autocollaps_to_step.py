"""add autocollaps to step

Revision ID: c8d7773b2b0b
Revises: 3fbc9c677acd
Create Date: 2026-09-02 22:17:49.797696

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'c8d7773b2b0b'
down_revision: Union[str, Sequence[str], None] = '3fbc9c677acd'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.execute('ALTER TABLE steps ADD COLUMN "autoCollapse" BOOLEAN;')

    


def downgrade() -> None:
    """Downgrade schema."""
    pass
