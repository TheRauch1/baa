import torch

# Define integer tensors
a = torch.tensor([1, 2, 3, 4, -1, -2, -3, -4], dtype=torch.int32)
b = torch.tensor([1, 0, 2, 3, 1, 2, 0, 3], dtype=torch.int32)

print("Tensor a:", a)
print("Tensor b:", b)

# Perform bitwise left shift
left_shift = torch.bitwise_left_shift(a, b)
print("\nLeft Shift Result (a << b):", left_shift)

# Perform bitwise right shift
right_shift = torch.bitwise_right_shift(a, b)
print("\nRight Shift Result (a >> b):", right_shift)

# Function to format binary representation of signed integers
def to_binary(val):
    return format(val & 0xffffffff, '032b')

# Explain the results
print("\nDetailed Explanation:")
for i in range(len(a)):
    a_val = a[i].item()
    b_val = b[i].item()
    ls_val = left_shift[i].item()
    rs_val = right_shift[i].item()

    print(f"Index {i}:")
    print(f"a[{i}] = {a_val} (binary: {to_binary(a_val)})")
    print(f"b[{i}] = {b_val}")

    print(f"Left Shift:")
    print(f"{a_val} << {b_val} = {ls_val} (binary: {to_binary(ls_val)})")

    print(f"Right Shift:")
    print(f"{a_val} >> {b_val} = {rs_val} (binary: {to_binary(rs_val)})\n")
