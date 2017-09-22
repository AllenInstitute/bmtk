from base_schema import PropertySchema as BaseSchema


class PropertySchema(BaseSchema):
    def get_params_column(self):
        return 'dynamics_params'
