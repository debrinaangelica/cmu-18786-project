
import matplotlib.pyplot as plt


def plot_loss(losses, x_axis_values=None, title='Training Loss', save_name='plot_loss.png'):
    plt.figure()
    if x_axis_values is not None:
        plt.plot(x_axis_values, losses, label=title)
    else:
        plt.plot(losses, label=title)
    plt.xlabel('epoch')
    plt.ylabel('mae loss')
    plt.legend()
    plt.grid(True)
    title = f"{title}\n(epoch: {len(losses)}, loss: {round(losses[-1], 2)})"
    plt.title(title)
    # plt.show()
    plt.savefig(save_name)
    plt.close()

def plot_predictions(y_true, y_pred, test_rmse, save_name='plot_predictions.png'):
    plt.figure()
    plt.plot(y_true, label="percent change")
    plt.plot(y_pred, label="predicted percent change")
    plt.xlabel('time')
    plt.ylabel('percent change')
    plt.legend()
    plt.grid(True)
    title = f"Test Predictions vs. True Stock Percent Change F(rmse={test_rmse})"
    plt.title(title)
    plt.savefig(save_name)
    # plt.show()

# TODO: WIP
# def plot_predictions_converted_to_close_prices(y_true_close, y_pred_percent_change, test_rmse):
#     """
#     Converts the % change predictions to closing prices
#     Plots these predicted closing prices against actual close prices of TSLA
#     """
#     plt.figure()
#     plt.plot(y_true_close, label="percent change")
#     y_pred_close = get_close_prices_from_percent_changes(y_true_close, y_pred_percent_change)
#     plt.plot(y_pred, label="predicted percent change")
#     plt.xlabel('time')
#     plt.ylabel('percent change')
#     plt.legend()
#     plt.grid(True)
#     title = f"Test Predictions vs. True Stock Percent Change (rmse={test_rmse})"
#     plt.title(title)
#     plt.show()