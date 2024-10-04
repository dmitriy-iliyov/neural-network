import matplotlib.pyplot as plt


def one_fit_statistic(data):
    # fig = plt.figure(figsize=(14, 10))
    fig = plt.figure(figsize=(12, 5))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1], height_ratios=[1], wspace=0.2, hspace=0.2)
    plt.subplots_adjust(left=0.05, right=0.98, bottom=0.1, top=0.95, wspace=0.2)

    ax1 = fig.add_subplot(gs[0])
    ax1.plot(range(data['epochs']), data['accuracy'], label='Accuracy', color='blue')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Accuracy')
    ax1.set_ylim(0, 1.1)
    ax1.legend()

    ax2 = fig.add_subplot(gs[1])
    ax2.plot(range(data['epochs']), data['loss'], label='Loss', color='red')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Loss')
    ax2.set_ylim(0, max(data['loss']) + 0.01)
    ax2.legend()

    hidden_layer_count = data.get('hidden_layer_count', 1)
    text_str = (f'Network: {data['network']}\n'
                f'Hidden layer count: {hidden_layer_count}\n'
                f'Hidden Neurons: {data["hidden_neurons_count"]}\n'
                f'Execution Time: {data["execution_time"]:.2f}s\n'
                f'Batch Size: {data["batch_size"]}')
    fig.text(0.315, 0.29, text_str, fontsize=11, verticalalignment='top',
             bbox=dict(facecolor='white', alpha=0.5))
    plt.show()


def draw_fig(x, y, z):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x, y, z, cmap='viridis')
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    plt.show()
