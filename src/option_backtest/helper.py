from datetime import datetime


def generate_output_dir(
    config: dict,
    output_dir: str,
    include_params=["stock_id", "s0", "mu", "sigma_stock", "start_date", "end_date"],
    exclude_params: set[str] = set(),
) -> str:
    output_dir_parts = []

    for param in include_params:
        if param not in exclude_params:
            value = config.get(param)
            if isinstance(value, datetime):
                value = value.strftime("%Y%m%d")
            elif isinstance(value, float):
                value = str(int(value * 100))
            output_dir_parts.append(f"{param}_{value}")

    # output_dir = (
    #     output_dir
    #     # + datetime.now().strftime("%Y%m%d_%H%M%S_")
    #     + "_".join(output_dir_parts)
    # )
    output_dir = f"{output_dir}/{'_'.join(output_dir_parts)}"

    return output_dir
