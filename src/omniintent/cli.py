"""
omniintent.cli
~~~~~~~~~~~~~~
Tiny CLI front-end for running the inference demo with *real* Quest\u00a03 sensor
logs.  Example:

    python -m omniintent demo --ingest samples/quest3_tiny.csv
"""
from pathlib import Path
import json
import torch
import typer

from omniintent.ingest.quest3_ingest import load as load_q3

__all__ = ["app"]

app = typer.Typer(add_completion=False, help="OmniIntent command-line tools")


@app.callback()
def main() -> None:
    """Main entry point for OmniIntent CLI."""
    pass

@app.command()
def demo(
    ingest: Path = typer.Option(
        None,
        "--ingest",
        "-i",
        exists=True,
        readable=True,
        help="Quest\u00a03 log file (.csv or .parquet). "
             "If omitted, synthetic data will be used.",
    ),
    seq_len: int = 60,
):
    """
    Run the transformer demo.  When `--ingest` is supplied the CSV/Parquet is
    fed into the model instead of synthetic batches.
    """
    if ingest is None:
        typer.secho("\u26A0\uFE0F  No --ingest provided \u2014 falling back to synthetic batch",
                    fg=typer.colors.YELLOW)
        try:
            # defer import so repo runs even if model code is absent
            from omniintent.model import MultiModalTransformer  # type: ignore
            batch = MultiModalTransformer.synthetic_batch(seq_len=seq_len)
            model = MultiModalTransformer.from_pretrained()
            output = model(**batch)
            typer.echo(output)
        except ModuleNotFoundError:
            typer.secho("Synthetic path requires `omniintent.model` \u2014 "
                        "module not found. Nothing to do.", fg=typer.colors.RED)
    else:
        batch = load_q3(str(ingest), seq_len=seq_len)

        # -----------------------------------------------------------------
        # Run the real transformer when available; otherwise fall back to
        # printing tensor shapes so developer envs without the heavyweight
        # model still work.
        # -----------------------------------------------------------------
        try:
            from omniintent.model import MultiModalTransformer  # type: ignore

            model = MultiModalTransformer.from_pretrained()
            model.eval()
            with torch.no_grad():
                output = model(**batch)

            io_shapes = {
                "input_shapes": {k: list(v.shape) for k, v in batch.items()},
                "output_shapes": {k: list(v.shape) for k, v in output.items()},
            }
            typer.echo(json.dumps(io_shapes, separators=(",", ":")))

        except Exception:
            typer.secho(
                "\u26A0\uFE0F  `omniintent.model.MultiModalTransformer` unavailable \u2014 "
                "printing input tensor shapes only.",
                fg=typer.colors.YELLOW,
                err=True,
            )
            shapes = {k: list(t.shape) for k, t in batch.items()}
            typer.echo(json.dumps({"input_shapes": shapes}, separators=(",", ":")))

if __name__ == "__main__":  # pragma: no cover
    app()
