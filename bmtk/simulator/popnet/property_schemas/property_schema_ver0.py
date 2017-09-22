from base_schema import PopTypes, PropertySchema as BaseSchema


class PropertySchema(BaseSchema):
    def get_params_column(self):
        return 'params_file'
