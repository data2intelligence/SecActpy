"""SecActpy interactive visualization app."""


def run_app(host: str = "127.0.0.1", port: int = 8050, debug: bool = False, **kwargs):
    """Launch the SecActpy Dash application.

    Parameters
    ----------
    host : str
        Host to bind to.
    port : int
        Port to serve on.
    debug : bool
        Enable Dash debug mode.
    """
    try:
        import dash  # noqa: F401
    except ImportError:
        raise ImportError(
            "Dash is required to run the SecActpy app.\n"
            "Install with: pip install secactpy[app]"
        )

    from secactpy.app.main import create_app

    app = create_app()
    app.run(host=host, port=port, debug=debug, **kwargs)
