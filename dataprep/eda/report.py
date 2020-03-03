from bokeh.models import LayoutDOM


class Report:
    to_render: LayoutDOM

    def __init__(self, to_render: LayoutDOM) -> None:
        self.to_render = to_render

    def save(self, path: str) -> None:
        pass

    def _repr_html_(self) -> str:
        return "<i>fill me</i>"
