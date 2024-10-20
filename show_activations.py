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
    register_linear_layer_forward_hook,
)
from baa.singletons import hidden_states

model_name = "meta-llama/Llama-3.2-1B-Instruct"
model = AutoModelForCausalLM.from_pretrained(model_name, device_map=device_map)
tokenizer = AutoTokenizer.from_pretrained(model_name)
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
benchmark = AccuracyBenchmark(model, tokenizer, dataset)

hist_data = []
keys = []

with torch.no_grad():
    add_custom_name_to_linear_layers(model)
    register_linear_layer_forward_hook(model, get_hidden_states_input)
    print("Original model accuracy:", benchmark.evaluate(sample_size=200))
    for key, value in hidden_states.items():
        value = torch.tensor(value).to("cpu")
        hist_data.append(value.numpy())
        keys.append(key)

fig = ff.create_distplot(
    hist_data,
    group_labels=keys,
    bin_size=0.1,
)

st.plotly_chart(fig, use_container_width=True)
