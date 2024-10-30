import os

import click
from rich import print
from rich.pretty import Pretty

from rag.config import appConfig
from rag.service.ollama_service import OllamaService
from rag.database import RagDb


@click.command()
@click.option(
    "--db",
    default=appConfig.get("DATABASE_PATH"),
    help="File path of the sqlite database to use.",
)
@click.option(
    "--schema",
    default=appConfig.get("SCHEMA_FILE"),
    help="The schema file used to create the database.",
)
def init_db(db, schema):
    """
    Will generate the sqlite database using the schema file.
    """
    RagDb.init_db(db, schema)


@click.command()
@click.option(
    "--db",
    default=appConfig.get("DATABASE_PATH"),
    help="File path of the sqlite database to drop.",
)
def drop_db(db="summarizer.db"):
    """Drop the database"""
    click.echo("Dropping the database ...")
    if os.path.isfile(db):
        os.remove(db)
        click.echo(f"Dropped the database:{os.path.abspath(db)}.")
    else:
        click.echo(f"Database {os.path.abspath(db)} not found.")


@click.command()
def config():
    """Dump the configuration."""
    print(Pretty(appConfig, expand_all=True))


@click.command()
@click.option("--model", default="llama3.1", help="The model used to chat.")
@click.option(
    "--prompt", default="Why is the sky blue?", help="The prompt used to chat."
)
@click.option("--role", default="user", help="The user type.")
def chat(model: str, prompt: str, role: str):
    """Get video text."""
    messages = [{"role": role, "content": prompt}]
    service = OllamaService()
    response = service.chat_with_model(model, messages)
    db = RagDb()
    db.insert_chat_response(response)
    print(response)


@click.group()
def cli():
    pass


cli.add_command(init_db)
cli.add_command(drop_db)
cli.add_command(config)
cli.add_command(chat)
