import os
from noitom.axis.mocap.base_extension import BaseExtension
from noitom.axis.mocap.client import MocapClient

class MocapClientExtension(BaseExtension):
    def on_startup(self, ext_id:str):
        print('MocapClientExtension.on_startup')
        super().on_startup(ext_id)
        super().start_extension(
            menu_name="",
            submenu_name="",
            name="Axis Mocap",
            title="Noitom Axis Mocap",
            doc_link = "",
            overview="",
            file_path = os.path.abspath(__file__),
            sample=MocapClient()
        )
        return