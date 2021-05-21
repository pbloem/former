import unittest
import torch

import transformers as trf
from former.util import slice_diag, compute_compression

class MyTestCase(unittest.TestCase):

    def test_slice_diagonal(self):

        m = torch.randint(high=20, size=(16, 24, 3, 5, 9))

        print(m[0, 0, 0])
        print(slice_diag(m[0, 0, 0], l=5))

    def test_gpt2(self):
        """
        Test the compute_compression function by checking the performance of GPT-2
        :return:
        """
        name = 'distilgpt2'

        tokenizer = trf.GPT2Tokenizer.from_pretrained(name)
        model = trf.GPT2LMHeadModel.from_pretrained(name)

        text = """
        George Walker Bush (born July 6, 1946) is an American politician and businessman who served as the 43rd president of the United States from 2001 to 2009. A member of the Republican Party, Bush previously served as the 46th governor of Texas from 1995 to 2000. He was born into the Bush family; his father, George H. W. Bush, was the 41st president of the United States from 1989 to 1993.
        """

        #
        # As the eldest son of Barbara and George H. W. Bush, he is the second son of a former United States president to himself become president, with the first being John Quincy Adams, the son of John Adams. He flew warplanes in the Texas and Alabama Air National Guard. After graduating from Yale College in 1968 and Harvard Business School in 1975, he worked in the oil industry. Bush married Laura Welch in 1977 and unsuccessfully ran for the U.S. House of Representatives shortly thereafter. He later co-owned the Texas Rangers baseball team before defeating incumbent Ann Richards in the 1994 Texas gubernatorial election. As governor, Bush successfully sponsored legislation for tort reform, increased education funding, set higher standards for schools, and reformed the criminal justice system. Bush also helped make Texas the leading producer of wind powered electricity in the U.S. Bush was elected president in 2000 when he defeated Democratic incumbent Vice President Al Gore after a narrow and contested win that involved a Supreme Court decision to stop a recount in Florida. He became the fourth person to be elected president without a popular vote victory.
        #
        # Upon taking office, Bush pushed through a $1.3 trillion tax cut program and the No Child Left Behind Act, a major education reform bill. He also pushed for socially conservative efforts, such as the Partial-Birth Abortion Ban Act and faith-based welfare initiatives. In response to the September 11 terrorist attacks, Bush created the Department of Homeland Security and launched a "War on Terror" that began with the war in Afghanistan in 2001. He also signed into law the controversial Patriot Act in order to authorize surveillance of suspected terrorists. In 2003, Bush ordered an invasion of Iraq, beginning the Iraq War, with his administration arguing that the Saddam Hussein regime possessed an active weapons of mass destruction (WMD) program, and that the Iraqi government posed a threat to the U.S. Some administration officials falsely claimed that Hussein had an operational relationship with Al-Qaeda, the perpetrators of the 9/11 attack. No stockpiles of WMDs or an active WMD program were ever found in Iraq. Bush also signed into law the Medicare Modernization Act, which created Medicare Part D, and funding for the AIDS relief program known as PEPFAR.
        #
        # Bush was re-elected to a second term in the 2004 presidential election, defeating Democratic Senator John Kerry in a close race. During his second term, Bush reached multiple free trade agreements and successfully nominated John Roberts and Samuel Alito to the Supreme Court. He sought major changes to Social Security and immigration laws, but both efforts failed. The wars in Afghanistan and Iraq continued, and in 2007 he launched a surge of troops in Iraq. Bush received criticism from across the political spectrum[4][5] for his handling of Hurricane Katrina[6][7] and for the midterm dismissal of U.S. attorneys. In the midst of it, the Democratic Party regained control of Congress in the 2006 elections. In December 2007, the U.S. entered the Great Recession, prompting the Bush administration to obtain congressional approval for multiple economic programs intended to preserve the country's financial system, including the Troubled Asset Relief Program (TARP) to buy toxic assets from financial institutions.
        #
        # Bush was among the most popular, as well as unpopular, U.S. presidents in history; he received the highest recorded approval ratings in the wake of the 9/11 attacks, but one of the lowest such ratings during the 2008 financial crisis.[8] Bush finished his second term in office in 2009 and returned to Texas. In 2010, he published his memoir, Decision Points.[9] His presidential library opened in 2013. His presidency has been rated as below-average in historical rankings of U.S. presidents, although his public favorability ratings have improved since leaving office.
        # """

        numchars = len(text)
        print(f'{numchars} characters in total.')
        encoded_input = tokenizer(text, return_tensors='pt')['input_ids'][0]

        if torch.cuda.is_available():
            model.cuda()

        bits = compute_compression(model, data=encoded_input, context=model.config.n_ctx, batch_size=32, verbose=True)

        print('total bits: ', bits)
        print('bits per byte: ', bits/numchars)

if __name__ == '__main__':
    unittest.main()
