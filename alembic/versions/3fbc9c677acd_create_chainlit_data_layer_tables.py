"""create chainlit data layer tables

Revision ID: 3fbc9c677acd
Revises:
Create Date: 2026-09-02 12:51:28.684267

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "3fbc9c677acd"
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.execute("""
    CREATE TABLE users (
      "id" UUID PRIMARY KEY,
      "identifier" TEXT NOT NULL UNIQUE,
      "metadata" JSONB NOT NULL,
      "createdAt" TEXT
    );
    """)

    op.execute("""
    CREATE TABLE IF NOT EXISTS threads (
      "id" UUID PRIMARY KEY,
      "createdAt" TEXT,
      "name" TEXT,
      "userId" UUID,
      "userIdentifier" TEXT,
      "tags" TEXT[],
      "metadata" JSONB,
      FOREIGN KEY ("userId") REFERENCES users("id") ON DELETE CASCADE
    );
    """)
    op.execute("""
    CREATE TABLE IF NOT EXISTS steps (
      "id" UUID PRIMARY KEY,
      "name" TEXT NOT NULL,
      "type" TEXT NOT NULL,
      "threadId" UUID NOT NULL,
      "parentId" UUID,
      "streaming" BOOLEAN NOT NULL,
      "waitForAnswer" BOOLEAN,
      "isError" BOOLEAN,
      "metadata" JSONB,
      "tags" TEXT[],
      "input" TEXT,
      "output" TEXT,
      "createdAt" TEXT,
      "start" TEXT,
      "end" TEXT,
      "generation" JSONB,
      "showInput" TEXT,
      "language" TEXT,
      "indent" INT,
      "defaultOpen" BOOLEAN,
      "modes" JSONB,
      FOREIGN KEY ("threadId") REFERENCES threads("id") ON DELETE CASCADE
    );
    """)
    op.execute("""
    CREATE TABLE IF NOT EXISTS elements (
      "id" UUID PRIMARY KEY,
      "threadId" UUID,
      "type" TEXT,
      "url" TEXT,
      "chainlitKey" TEXT,
      "name" TEXT NOT NULL,
      "display" TEXT,
      "objectKey" TEXT,
      "size" TEXT,
      "page" INT,
      "language" TEXT,
      "forId" UUID,
      "mime" TEXT,
      FOREIGN KEY ("threadId") REFERENCES threads("id") ON DELETE CASCADE
    );
    """)
    op.execute("""
    CREATE TABLE IF NOT EXISTS feedbacks (
      "id" UUID PRIMARY KEY,
      "forId" UUID NOT NULL,
      "threadId" UUID NOT NULL,
      "value" INT NOT NULL,
      "comment" TEXT,
      FOREIGN KEY ("threadId") REFERENCES threads("id") ON DELETE CASCADE
    );
    """)


def downgrade() -> None:
    """Downgrade schema."""
    pass
