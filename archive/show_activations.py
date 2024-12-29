import numpy as np
import plotly.figure_factory as ff
import streamlit as st
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from baa import (
    AccuracyBenchmark,
    add_custom_name_to_linear_layers,
    device_map,
    get_hidden_states_input,
    get_hidden_states_output,
    register_linear_layer_forward_hook,
    remove_all_hooks,
)
from baa.singletons import hidden_states

model_name = "meta-llama/Llama-3.2-1B-Instruct"
model = AutoModelForCausalLM.from_pretrained(model_name, device_map=device_map)
tokenizer = AutoTokenizer.from_pretrained(model_name)
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
benchmark = AccuracyBenchmark(model, tokenizer, dataset)

hist_data_input = []
hist_data_output = []
keys_input = []
keys_output = []

with torch.no_grad():
    add_custom_name_to_linear_layers(model)
    register_linear_layer_forward_hook(model, get_hidden_states_input)
    print("Original model accuracy:", benchmark.evaluate(sample_size=200))
    hidden_states_input = hidden_states.copy()
    hidden_states.clear()
    remove_all_hooks(model)
    register_linear_layer_forward_hook(model, get_hidden_states_output)
    print("Original model accuracy:", benchmark.evaluate(sample_size=200))
    for key, value in hidden_states_input.items():
        value = torch.tensor(value).to("cpu")
        hist_data_input.append(value.numpy())
        keys_input.append(key)

    # for key, value in hidden_states.items():
    #     value = torch.tensor(value).to("cpu")
    #     hist_data_output.append(value.numpy())
    #     keys_output.append(key)

fig_input = ff.create_distplot(
    hist_data_input,
    group_labels=keys_input,
    bin_size=0.1,
)

# fig_output = ff.create_distplot(
#     hist_data_output,
#     group_labels=keys_output,
#     bin_size=0.1,
# )

st.title("Activations")

st.subheader("Inputs")
st.plotly_chart(fig_input, use_container_width=True)

st.subheader("Outputs")
# st.plotly_chart(fig_output, use_container_width=True)
