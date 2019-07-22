from config import config
import loader
import trainer

dataloader = loader.CelebA_Dataloader(config.data_csv_path, config.image_path,
                                      config.targets, config.batch_size,
                                      config.worker_number,
                                      config.input_image_size)

trainer.Trainer(dataloader, config.number_of_epochs).train()
