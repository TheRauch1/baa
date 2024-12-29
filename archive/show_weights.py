import numpy as np
import plotly.figure_factory as ff
import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer

from baa import (
    QuantizedLinearLayerWithActivation,
    add_custom_name_to_linear_layers,
    device_map,
    get_weights,
    replace_linear_layer_with_activation,
)
from baa.singletons import hidden_states

model_name = "HuggingFaceTB/SmolLM-135M"
# model_name = "meta-llama/Llama-3.2-1B-Instruct"
model = AutoModelForCausalLM.from_pretrained(model_name, device_map=device_map)
tokenizer = AutoTokenizer.from_pretrained(model_name)

add_custom_name_to_linear_layers(model)
replace_linear_layer_with_activation(
    base_model=model,
    quantizer_class=QuantizedLinearLayerWithActivation,
    weight_bits=8,
    activation_bits=6,
    exclude_list=[],
    quantized=True,
)

weight_data = get_weights(model)
first_key = list(weight_data.keys())[0]
test = {}
test[first_key] = weight_data[first_key]
random_half_of_weight_data_dict = {}
# randomly select 50% of the keys and values and add those
for key, value in weight_data.items():
    if np.random.randint(10) == 0:
        random_half_of_weight_data_dict[key] = value
keys = []
weights = []
for key, weight in random_half_of_weight_data_dict.items():
    keys.append(key)
    weights.append(np.random.choice(weight.view(-1).cpu().numpy(), 100))


fig_input = ff.create_distplot(
    weights,
    group_labels=keys,
    bin_size=0.01,
)

st.title("Weights")

st.plotly_chart(fig_input, use_container_width=True)
