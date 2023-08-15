import json

from loguru import logger
from torch.utils.data import Dataset


# 这里定义的数据集的格式为:
#  每一行就是一个完整的json格式

class SFTDataset(Dataset):
    def __init__(self, file, tokenizer, max_seq_length):
        self.tokenizer = tokenizer
        self.bos_token_id = tokenizer.bos_token_id  # Begin of Sentence（BOS）的缩写，表示句子的开头 ,就是 训练词中的起始标记和结束标记
        self.eos_token_id = tokenizer.eos_token_id  # End of Sentence（EOS）的缩写，表示句子的结束
        self.max_seq_length = max_seq_length  # 最大的序列长度,qwen的配置文件上是1024
        logger.info('Loading data: {}'.format(file))
        print(f"起始标记和结束标记: bos: {self.bos_token_id}, eos: {self.eos_token_id} \n")
        with open(file, 'r', encoding='utf8') as f:
            data_list = f.readlines()
        # 读取数据集的所有数据到内存中,为一个列表,每一个元素即是一行文本内容
        logger.info("there are {} data in dataset".format(len(data_list)))
        self.data_list = data_list

    def __len__(self):
        # 返回数据集的数量大小,每一个完整的训练内容,就是一条记录
        return len(self.data_list)

    # 返回的SFTDataset 对象作为可迭代对象,当被调用方迭代数据集时,将迭代的索引传入,并返回一个索引对应的对象
    def __getitem__(self, index):
        # 每条数据格式为: <s>input1</s>target1</s>input2</s>target2</s>...
        data = self.data_list[index]
        # 解析一条训练数据为json对象
        data = json.loads(data)
        ## 一个完整的 conversation 示例
        # {"conversation_id": 36, "category": "Brainstorming", "conversation": [{"human": "如何在不影响电子产品性能的情况下最大限度地减少耗材成本？", "assistant": "要在不影响电子产品性能的情况下最大限度地减少耗材成本，可以考虑以下几个建议：\n\n1. 选择合适的耗材：选择优质和经济实惠的耗材，而不是只追求品牌或高端的耗材。例如，一些中低价位的耗材可能与高价位的耗材具有相似的性能，但价格更为实惠。\n\n2. 优化使用方法：使用正确的工具和方法，避免浪费耗材。例如，在使用胶水时，避免使用过多的胶水，以减少浪费。\n\n3. 掌握存储技巧：正确地储存耗材可以延长其寿命，减少浪费。例如，在存储纸张时，应将其放在干燥、清洁的地方，避免受潮或受到污染。\n\n4. 找到更好的供应商：与不同的供应商合作，比较价格和质量，以获得更好的耗材价格。同时，也要确保供应商提供的耗材符合质量标准，并可靠。\n\n5. 建立合理的库存管理：建立合理的库存管理系统，以避免过多的耗材积压，浪费资金和资源。同时，也要确保有足够的耗材储备，以满足生产需要。"}, {"human": "你能给我一些关于库存管理的具体建议吗？", "assistant": "当涉及到库存管理时，以下是一些具体建议：\n\n1. 了解需求：了解您的产品销售量和预测，以便确定合适的库存水平。使用数据分析和预测工具可帮助您更准确地预测需求。\n\n2. 建立库存管理系统：建立一个全面的库存管理系统，以跟踪库存水平、库存周转率、库存成本和库存损失等信息。使用库存管理软件可以帮助您更轻松地管理库存，并优化库存供应链。\n\n3. 优化供应链：与供应商进行合理的谈判，以获得更好的库存价格和减少库存积压。同时，要确保及时交付和良好的供应商关系，以确保库存水平的稳定。\n\n4. 使用先进的技术：使用先进的技术，例如条形码、RFID等，以帮助您更轻松地跟踪库存，并更好地管理库存。这些技术可以帮助您更准确地估算库存水平，减少库存积压和浪费。\n\n5. 定期审查库存：定期审查库存，以确定哪些库存需要重新定购，哪些库存需要清理。这有助于减少库存积压和减少库存损失。\n\n6. 确保库存安全：为了确保库存安全，请采取必要的安全措施，并定期检查库存以确保库存未受到损害或被盗窃。"}, {"human": "在建立库存管理系统时，你能否推荐一些可用的软件或工具？", "assistant": "当建立库存管理系统时，以下是一些可用的软件或工具：\n\n1. QuickBooks：QuickBooks是一款财务管理软件，提供库存追踪和管理工具，可帮助您跟踪库存水平、库存成本和库存周转率等信息。\n\n2. TradeGecko：TradeGecko是一款云端库存管理软件，提供库存追踪、采购管理和销售订单处理等功能，可帮助您优化库存供应链和提高库存效率。\n\n3. Fishbowl：Fishbowl是一款基于QuickBooks的库存管理软件，提供库存追踪、采购管理、销售订单处理和制造管理等功能，可帮助您全面管理库存。\n\n4. Zoho Inventory：Zoho Inventory是一款云端库存管理软件，提供库存追踪、采购管理、销售订单处理和发货管理等功能，可帮助您全面管理库存供应链。\n\n5. inFlow Inventory：inFlow Inventory是一款基于云端的库存管理软件，提供库存追踪、采购管理、销售订单处理和报告等功能，可帮助您更好地管理库存和提高库存效率。\n\n这些库存管理软件都具有不同的功能和优势，您可以根据自己的需求和预算选择最适合您的软件或工具。"}, {"human": "你能否给我一些关于使用RFID技术来管理库存的具体建议？", "assistant": "当使用RFID技术来管理库存时，以下是一些具体建议：\n\n1. 确定RFID标签的类型：根据库存管理需求和预算，选择适当的RFID标签类型。有些标签只能读取一次，而其他标签则可以重复读取。\n\n2. 部署RFID阅读器：部署RFID阅读器以读取标签，并确保阅读器与标签兼容。如果需要覆盖更大的区域，则需要安装更多的阅读器。\n\n3. 标记库存：将RFID标签附加到库存上，并记录每个标签的位置和其他相关信息。这可以帮助您更好地追踪库存并减少库存积压。\n\n4. 确保RFID系统的准确性：确保RFID系统准确读取标签并正确记录库存。使用校准工具定期检查RFID阅读器的准确性。\n\n5. 整合RFID系统：整合RFID系统与其他库存管理系统（例如ERP软件）以更好地管理库存。这可以帮助您更快地识别库存问题，并更快地做出决策。\n\n6. 培训员工：培训员工使用RFID技术，以确保他们了解如何正确地使用RFID标签和阅读器，并能够正确地记录和追踪库存。\n\n
        # 7. 定期维护RFID系统：定期维护RFID系统，以确保阅读器和标签的正常工作。更换损坏的标签和阅读器，并定期更新RFID系统软件。"}], "dataset": "moss"}
        conversation = data['conversation']

        # 收集多轮对话
        utterances = []
        for x in conversation:
            utterances.append(x['human'])
            utterances.append(x['assistant'])
        #     将utterances列表中的话语使用分词器进行编码，得到utterances_ids，表示分词后的标记ID。
        utterances_ids = self.tokenizer(utterances, add_special_tokens=False).input_ids

        print(f"input_ids tokenizer of transdata: {utterances_ids} \n")

        # 模型的输入格式为：<s>input1</s>target1</s>input2</s>target2</s>...
        input_ids = [self.bos_token_id]
        target_mask = [0]  # 用于对input进行mask，只计算target部分的loss
        for i, utterances_id in enumerate(utterances_ids):
            input_ids += (utterances_id + [self.eos_token_id])
            if i % 2 == 0:
                target_mask += [0] * (len(utterances_id) + 1)
            else:
                target_mask += [1] * (len(utterances_id) + 1)
        assert len(input_ids) == len(target_mask)
        # 对长度进行截断
        input_ids = input_ids[:self.max_seq_length]
        target_mask = target_mask[:self.max_seq_length]
        attention_mask = [1] * len(input_ids)
        assert len(input_ids) == len(target_mask) == len(attention_mask)
        inputs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'target_mask': target_mask
        }
        return inputs


class ChatGLM2SFTDataset(SFTDataset):

    def __getitem__(self, index):
        """
        基本沿袭ChatGLM2的指令微调的格式，做了小修改，多轮对话如下。
        """
        # 每条数据格式为: [Round 1]\n\n问：{input1}\n\n答：{target1}</s>[Round 2]\n\n问：{input2}\n\n答：{target2}</s>...
        data = self.data_list[index]
        data = json.loads(data)
        conversation = data['conversation']
        input_format = '[Round {}]\n\n问：{}\n\n答：'
        target_format = '{}'

        # 收集多轮对话
        utterances = []
        # enumerate 是 Python 内置的一个函数，用于在迭代（如列表、元组或字符串等）时同时获取元素的索引和值。
        # 它返回一个由索引和对应值组成的迭代器（iterator），可以通过循环来逐个访问这些索引和值。
        for i, x in enumerate(conversation):
            human = input_format.format(i + 1, x['human'])
            assistant = target_format.format(x['assistant'])
            utterances += ([human, assistant])
        utterances_ids = self.tokenizer(utterances, add_special_tokens=False).input_ids

        # 每条数据格式为: [Round 1]\n\n问：{input1}\n\n答：{target1}</s>[Round 2]\n\n问：{input2}\n\n答：{target2}</s>...
        input_ids = []
        target_mask = []  # 用于对input进行mask，只计算target部分的loss
        for i, utterances_id in enumerate(utterances_ids):
            input_ids += utterances_id
            # input部分
            if i % 2 == 0:
                target_mask += [0] * (len(utterances_id))
            # target部分
            else:
                input_ids += [self.eos_token_id]
                target_mask += [1] * (len(utterances_id) + 1)
        assert len(input_ids) == len(target_mask)
        # 对长度进行截断
        input_ids = input_ids[:self.max_seq_length]
        target_mask = target_mask[:self.max_seq_length]
        attention_mask = [1] * len(input_ids)
        assert len(input_ids) == len(target_mask) == len(attention_mask)
        inputs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'target_mask': target_mask
        }
        return inputs
