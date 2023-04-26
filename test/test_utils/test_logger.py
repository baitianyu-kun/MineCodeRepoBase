import utils.logger as logger
import test_logger2


logger, log_path = logger.prepare_logger('./logs_test', 'baitianyu')
logger.info("helloworld")
test_logger2.test()

