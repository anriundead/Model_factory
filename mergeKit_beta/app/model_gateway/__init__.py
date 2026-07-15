"""Published-model gateway module entrypoint."""


def register_model_gateway(app):
    """Register published-model gateway routes."""
    from app.model_gateway.routes import model_gateway_bp

    app.register_blueprint(model_gateway_bp)
    return app
