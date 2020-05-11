macroWords = ['us', 
'usa', 
'united states', 
'united states of america', 
'fed', 
'treasury', 
'irs', 
'u.s.', 
'North America',
'trade', 
'interest rate', 
'fed', 
'gdp', 
'inflation', 
'cpi', 
'price index', 
'gnp',
'economy',
'economic',
'bond',
]

sectorWords = ['energy', 'materials', 'industrial', 'discretionary', 'staples', 'health', 'financial', 'technology', 'information technology', 'telecommucation', 'utilities', 'real estate']

tickerList = ['mmm', 'aos', 'abt', 'abbv', 'acn', 'atvi', 'ayi', 'adbe', 
'aap', 'amd', 'aes', 
'aet', 'amg', 'afl', 'a', 'apd', 'akam', 'alk', 'alb', 
'are', 'alxn', 'algn', 'alle', 'agn', 'ads', 'lnt', 'all', 
'googl', 'goog', 'mo', 'amzn', 'aee', 'aal', 'aep', 'axp', 
'aig', 'amt', 'awk', 'amp', 'abc', 'ame', 'amgn', 'aph', 'apc', 
'adi', 'andv', 'anss', 'antm', 'aon', 'apa', 'aiv', 'aapl', 'amat',
 'adm', 'arnc', 'ajg', 'aiz', 't', 'adsk', 'adp', 'azo', 'avb', 'avy', 
 'bhge', 'bll', 'bac', 'bax', 'bbt', 'bdx', 'bby', 'biib', 'blk', 'hrb', 'ba', 
 'bwa', 'bxp', 'bsx', 'bhf', 'bmy', 'avgo', 'chrw', 'ca', 'cog', 'cdns', 'cpb', 'cof', 'cah',
  'kmx', 'ccl', 'cat', 'cboe', 'cbs', 'celg', 'cnc', 'cnp', 'ctl', 'cern', 'cf', 
  'schw', 'chtr', 'cvx', 'cmg', 'cb', 'chd', 'ci', 'xec', 'cinf', 'ctas', 'csco', 'c',
  'cfg', 'ctxs', 'cme', 'cms', 'ko', 'ctsh', 'cl', 'cmcsa', 'cma', 'cag', 'cxo', 'cop', 
  'ed', 'stz', 'glw', 'cost', 'coty', 'cci', 'csra', 'csx', 'cmi', 'cvs', 'dhi', 'dhr', 
  'dri', 'dva', 'de', 'dal', 'xray', 'dvn', 'dlr', 'dfs', 'disca', 'disck', 'dish',
   'dg', 'dltr', 'd', 'dov', 'dwdp', 'dps', 'dte', 'duk', 'dre', 'dxc', 'etfc', 'emn',
    'etn', 'ebay', 'ecl', 'eix', 'ew', 'ea', 'emr', 'etr', 'evhc', 'eog', 'eqt', 'efx',
    'eqix', 'eqr', 'ess', 'el', 're', 'es', 'exc', 'expe', 'expd', 'esrx', 'exr', 'xom',
     'ffiv', 'fb', 'fast', 'frt', 'fdx', 'fis', 'fitb', 'fe', 'fisv', 'flir', 'fls', 'flr', 
     'fmc', 'fl', 'f', 'ftv', 'fbhs', 'ben', 'fcx', 'gps', 'grmn', 'it', 'gd', 'ge', 'ggp', 
     'gis', 'gm', 'gpc', 'gild', 'gpn', 'gs', 'gt', 'gww', 'hal', 'hbi', 'hog', 'hrs', 
     'hig', 'has', 'hca', 'hcp', 'hp', 'hsic', 'hes', 'hpe', 'hlt', 'holx', 'hd', 'hon',
      'hrl', 'hst', 'hpq', 'hum', 'hban', 'hii', 'idxx', 'info', 'itw', 'ilmn', 'incy', 
      'ir', 'intc', 'ice', 'ibm', 'ip', 'ipg', 'iff', 'intu', 'isrg', 'ivz', 'ipgp', 'irm', 'jbht', 'jec', 'sjm', 'jnj', 'jci', 'jpm', 'jnpr', 'ksu', 'k', 'key', 'kmb', 'kim', 'kmi', 'klac', 'kss', 'khc', 'kr', 'lb', 'lll', 'lh', 'lrcx', 'leg', 'len', 'luk', 'lly', 'lnc', 'lkq', 'lmt', 'l', 'low', 'lyb', 'mtb', 'mac', 'm', 'mro', 'mpc', 'mar', 'mmc', 'mlm', 'mas', 'ma', 'mat', 'mkc', 'mcd', 'mck', 'mdt', 'mrk', 'met', 'mtd', 'mgm', 'kors', 'mchp', 'mu', 'msft', 'maa', 'mhk', 'tap', 'mdlz', 'mon', 'mnst', 'mco', 'ms', 'msi', 'myl', 'ndaq', 'nov', 'navi', 'nktr', 'ntap', 'nflx', 'nwl', 'nfx', 'nem', 'nwsa', 'nws', 'nee', 'nlsn', 'nke', 'ni', 'nbl', 'jwn', 'nsc', 'ntrs', 'noc', 'nclh', 'nrg', 'nue', 'nvda', 'orly', 'oxy', 'omc', 'oke', 'orcl', 'pcar', 'pkg', 'ph', 'payx', 'pypl', 'pnr', 'pbct', 'pep', 'pki', 'prgo', 'pfe', 'pcg', 'pm', 'psx', 'pnw', 'pxd', 'pnc', 'rl', 'ppg', 'ppl', 'px', 'pfg', 'pg', 'pgr', 'pld', 'pru', 'peg', 'psa', 'phm', 'pvh', 'qrvo', 'qcom', 'pwr', 'dgx', 'rrc', 'rjf', 'rtn', 'o', 'rht', 'reg', 'regn', 'rf', 'rsg', 'rmd', 'rhi', 'rok', 'col', 'rop', 'rost', 'rcl', 'spgi', 'crm', 'sbac', 'scg', 'slb', 'stx', 'see', 'sre', 'shw', 'spg', 'swks', 'slg', 'sna', 'so', 'luv', 'swk', 'sbux', 'stt', 'srcl', 'syk', 'sti', 'sivb', 'symc', 'syf', 'snps', 'syy', 'trow', 'ttwo', 'tpr', 'tgt', 'tel', 'fti', 'txn', 'txt', 'bk', 'clx', 'coo', 'hsy', 'mos', 'trv', 'dis', 'tmo', 'tif', 'twx', 'tjx', 'tmk', 'tss', 'tsco', 'tdg', 'trip', 'foxa', 'fox', 'tsn', 'usb', 'udr', 'ulta', 'uaa', 'ua', 'unp', 'ual', 'unh', 'ups', 'uri', 'utx', 'uhs', 'unm', 'vfc', 'vlo', 'var', 
'vtr', 'vrsn', 'vrsk', 'vz', 'vrtx', 'viab', 'v', 'vno', 'vmc', 'wmt', 'wba', 'wm',
 'wat', 'wec', 'wfc', 'wdc', 'wu', 'wrk', 'wy', 'whr', 'wmb', 'wltw', 'wyn', 'wynn',
  'xel', 'xrx', 'xlnx', 'xl', 'xyl', 'yum', 'zbh', 'zion', 'zts']

companies = {'3m company': 'mmm', 'a.o. smith corp': 'aos', 
'abbott laboratories': 'abt', 'abbvie inc.': 'abbv', 
'accenture plc': 'acn', 'activision blizzard': 'atvi', 
'acuity brands inc': 'ayi', 'adobe systems inc': 'adbe', 
'advance auto parts': 'aap', 'advanced micro devices inc': 'amd',
'aes corp': 'aes', 'aetna inc': 'aet', 
'affiliated managers group inc': 'amg', 'aflac inc': 'afl', 
'agilent technologies inc': 'a', 'air products & chemicals inc': 'apd',
'akamai technologies inc': 'akam', 'alaska air group inc': 'alk', 
'albemarle corp': 'alb', 'alexandria real estate equities inc': 'are',
'alexion pharmaceuticals': 'alxn', 'align technology': 'algn',
'allegion': 'alle', 'allergan, plc': 'agn',
'alliance data systems': 'ads', 'alliant energy corp': 'lnt',
'allstate corp': 'all', 'alphabet inc class a': 'googl', 
'alphabet inc class c': 'goog', 'altria group inc': 'mo', 
'amazon.com inc.': 'amzn', 'ameren corp': 'aee', 
'american airlines group': 'aal', 'american electric power': 'aep',
'american express co': 'axp', 
'american international group, inc.': 'aig', 
'american tower corp a': 'amt', 
'american water works company inc': 'awk',
'ameriprise financial': 'amp', 'amerisourcebergen corp': 'abc',
'ametek inc.': 'ame', 'amgen inc.': 'amgn', 
'amphenol corp': 'aph', 'anadarko petroleum corp': 'apc',
'analog devices, inc.': 'adi', 'andeavor': 'andv', 'ansys':
'anss', 'anthem inc.': 'antm', 'aon plc': 'aon',
'apache corporation': 'apa', 
'apartment investment & management': 'aiv', 
'apple inc.': 'aapl', 'applied materials inc.': 'amat',
'archer-daniels-midland co': 'adm', 'arconic inc.': 'arnc',
'arthur j. gallagher & co.': 'ajg', 'assurant inc.': 'aiz',
'at&t inc.': 't', 'autodesk inc.': 'adsk', 
'automatic data processing': 'adp', 'autozone inc': 'azo', 
'avalonbay communities, inc.': 'avb', 'avery dennison corp': 'avy', 
'baker hughes, a ge company': 'bhge', 'ball corp': 'bll', 
'bank of america corp': 'bac', 'baxter international inc.': 'bax',
'bb&t corporation': 'bbt', 'becton dickinson': 'bdx', 
'best buy co. inc.': 'bby', 'biogen inc.': 'biib', 'blackrock': 'blk', 
'block h&r': 'hrb', 'boeing company': 'ba', 'borgwarner': 'bwa', 
'boston properties': 'bxp', 'boston scientific': 'bsx', 
'brighthouse financial inc': 'bhf', 'bristol-myers squibb': 'bmy',
'broadcom': 'avgo', 'c. h. robinson worldwide': 'chrw', 'ca, inc.': 'ca',
'cabot oil & gas': 'cog', 'cadence design systems': 'cdns', 
'campbell soup': 'cpb', 'capital one financial': 'cof', 
'cardinal health inc.': 'cah', 'carmax inc': 'kmx', 'carnival corp.': 'ccl',
'caterpillar inc.': 'cat', 'cboe global markets': 'cboe', 
'cbs corp.': 'cbs', 'celgene corp.': 'celg', 'centene corporation': 'cnc',
'centerpoint energy': 'cnp', 
'centurylink inc': 'ctl', 'cerner': 'cern', 
'cf industries holdings inc': 'cf', 'charles schwab corporation': 'schw', 
'charter communications': 'chtr', 'chevron corp.': 'cvx', 
'chipotle mexican grill': 'cmg', 'chubb limited': 'cb', 
'church & dwight': 'chd', 'cigna corp.': 'ci', 'cimarex energy': 'xec', 
'cincinnati financial': 'cinf', 'cintas corporation': 'ctas', 
'cisco systems': 'csco', 'citigroup inc.': 'c', 
'citizens financial group': 'cfg', 'citrix systems': 'ctxs', 
'cme group inc.': 'cme', 'cms energy': 'cms',
'coca-cola company (the)': 'ko', 'cognizant technology solutions': 'ctsh', 
'colgate-palmolive': 'cl', 'comcast corp.': 'cmcsa', 'comerica inc.': 'cma',
'conagra brands': 'cag', 'concho resources': 'cxo', 'conocophillips': 'cop',
'consolidated edison': 'ed', 'constellation brands': 'stz',
'corning inc.': 'glw', 'costco wholesale corp.': 'cost',
'coty, inc': 'coty', 'crown castle international corp.': 'cci',
'csra inc.': 'csra', 'csx corp.': 'csx', 'cummins inc.': 'cmi', 
'cvs health': 'cvs', 'd. r. horton': 'dhi', 'danaher corp.': 'dhr', 
'darden restaurants': 'dri', 'davita inc.': 'dva', 'deere & co.': 'de',
'delta air lines inc.': 'dal', 'dentsply sirona': 'xray',
'devon energy corp.': 'dvn', 'digital realty trust inc': 'dlr',
'discover financial services': 'dfs',
'discovery inc. class a': 'disca',
'discovery inc. class c': 'disck', 'dish network': 'dish',
'dollar general': 'dg', 'dollar tree': 'dltr', 
'dominion energy': 'd', 'dover corp.': 'dov',
'dowdupont': 'dwdp', 'dr pepper snapple group': 'dps', 
'dte energy co.': 'dte', 'duke energy': 'duk', 
'duke realty corp': 'dre', 'dxc technology': 'dxc', 
'e*trade': 'etfc', 'eastman chemical': 'emn',
'eaton corporation': 'etn', 'ebay inc.': 'ebay',
'ecolab inc.': 'ecl', "edison int'l": 'eix',
'edwards lifesciences': 'ew', 'electronic arts': 'ea', 
'emerson electric company': 'emr', 'entergy corp.': 'etr',
'envision healthcare': 'evhc', 'eog resources': 'eog', 
'eqt corporation': 'eqt', 'equifax inc.': 'efx', 
'equinix': 'eqix', 'equity residential': 'eqr', 
'essex property trust, inc.': 'ess', 
'estee lauder cos.': 'el', 'everest re group ltd.': 're', 
'eversource energy': 'es', 'exelon corp.': 'exc', 'expedia inc.': 'expe', 
'expeditors international': 'expd', 'express scripts': 'esrx', 
'extra space storage': 'exr', 'exxon mobil corp.': 'xom',
'f5 networks': 'ffiv', 'facebook, inc.': 'fb', 'fastenal co': 'fast', 
'federal realty investment trust': 'frt', 'fedex corporation': 'fdx',
'fidelity national information services': 'fis', 
'fifth third bancorp': 'fitb', 'firstenergy corp': 'fe', 
'fiserv inc': 'fisv', 'flir systems': 'flir', 
'flowserve corporation': 'fls', 'fluor corp.': 'flr', 
'fmc corporation': 'fmc', 'foot locker inc': 'fl', 'ford motor': 'f', 
'fortive corp': 'ftv', 'fortune brands home & security': 'fbhs', 
'franklin resources': 'ben', 'freeport-mcmoran inc.': 'fcx', 
'gap inc.': 'gps', 'garmin ltd.': 'grmn', 'gartner inc': 'it', 
'general dynamics': 'gd', 'general electric': 'ge', 
'general growth properties inc.': 'ggp', 'general mills': 'gis', 
'general motors': 'gm', 'genuine parts': 'gpc', 'gilead sciences': 'gild', 'global payments inc.': 'gpn', 'goldman sachs group': 'gs', 'goodyear tire & rubber': 'gt', 'grainger (w.w.) inc.': 'gww', 'halliburton co.': 'hal', 'hanesbrands inc': 'hbi', 'harley-davidson': 'hog', 'harris corporation': 'hrs', 'hartford financial svc.gp.': 'hig', 'hasbro inc.': 'has', 'hca holdings': 'hca', 'hcp inc.': 'hcp', 'helmerich & payne': 'hp', 'henry schein': 'hsic', 'hess corporation': 'hes', 'hewlett packard enterprise': 'hpe', 'hilton worldwide holdings inc': 'hlt', 'hologic': 'holx', 'home depot': 'hd', "honeywell int'l inc.": 'hon', 'hormel foods corp.': 'hrl', 'host hotels & resorts': 'hst', 'hp inc.': 'hpq', 'humana inc.': 'hum', 'huntington bancshares': 'hban', 'huntington ingalls industries': 'hii', 'idexx laboratories': 'idxx', 'ihs markit ltd.': 'info', 'illinois tool works': 'itw', 'illumina inc': 'ilmn', 'incyte': 'incy', 'ingersoll-rand plc': 'ir', 'intel corp.': 'intc', 'intercontinental exchange': 'ice', 'international business machines': 'ibm', 'international paper': 'ip', 'interpublic group': 'ipg', 'intl flavors & fragrances': 'iff', 'intuit inc.': 'intu', 'intuitive surgical inc.': 'isrg', 'invesco ltd.': 'ivz', 'ipg photonics corp.': 'ipgp', 'iron mountain incorporated': 'irm', 'j. b. hunt transport services': 'jbht', 'jacobs engineering group': 'jec', 'jm smucker': 'sjm', 'johnson & johnson': 'jnj', 'johnson controls international': 'jci', 'jpmorgan chase & co.': 'jpm', 'juniper networks': 'jnpr', 'kansas city southern': 'ksu', 'kellogg co.': 'k', 'keycorp': 'key', 'kimberly-clark': 'kmb', 'kimco realty': 'kim', 'kinder morgan': 'kmi', 'kla-tencor corp.': 'klac', "kohl's corp.": 'kss', 'kraft heinz co': 'khc', 'kroger co.': 'kr', 'l brands inc.': 'lb', 'l-3 communications holdings': 'lll', 'laboratory corp. of america holding': 'lh', 'lam research': 'lrcx', 'leggett & platt': 'leg', 'lennar corp.': 'len', 'leucadia national corp.': 'luk', 'lilly (eli) & co.': 'lly', 'lincoln national': 'lnc', 'lkq corporation': 'lkq', 'lockheed martin corp.': 'lmt', 'loews corp.': 'l', "lowe's cos.": 'low', 'lyondellbasell': 'lyb', 'm&t bank corp.': 'mtb', 'macerich': 'mac', "macy's inc.": 'm', 'marathon oil corp.': 'mro', 'marathon petroleum': 'mpc', "marriott int'l.": 'mar', 'marsh & mclennan': 'mmc', 'martin marietta materials': 'mlm', 'masco corp.': 'mas', 'mastercard inc.': 'ma', 'mattel inc.': 'mat', 'mccormick & co.': 'mkc', "mcdonald's corp.": 'mcd', 'mckesson corp.': 'mck', 'medtronic plc': 'mdt', 'merck & co.': 'mrk', 'metlife inc.': 'met', 'mettler toledo': 'mtd', 'mgm resorts international': 'mgm', 'michael kors holdings': 'kors', 'microchip technology': 'mchp', 'micron technology': 'mu', 'microsoft corp.': 'msft', 'mid-america apartments': 'maa', 'mohawk industries': 'mhk', 'molson coors brewing company': 'tap', 'mondelez international': 'mdlz', 'monsanto co.': 'mon', 'monster beverage': 'mnst', "moody's corp": 'mco', 'morgan stanley': 'ms', 'motorola solutions inc.': 'msi', 'mylan n.v.': 'myl', 'nasdaq, inc.': 'ndaq', 'national oilwell varco inc.': 'nov', 'navient': 'navi', 'nektar therapeutics': 'nktr', 'netapp': 'ntap', 'netflix inc.': 'nflx', 'newell brands': 'nwl', 'newfield exploration co': 'nfx', 'newmont mining corporation': 'nem', 'news corp. class a': 'nwsa', 'news corp. class b': 'nws', 'nextera energy': 'nee', 'nielsen holdings': 'nlsn', 'nike': 'nke', 'nisource inc.': 'ni', 'noble energy inc': 'nbl', 'nordstrom': 'jwn', 'norfolk southern corp.': 'nsc', 'northern trust corp.': 'ntrs', 'northrop grumman corp.': 'noc', 'norwegian cruise line': 'nclh', 'nrg energy': 'nrg', 'nucor corp.': 'nue', 'nvidia corporation': 'nvda', "o'reilly automotive": 'orly', 'occidental petroleum': 'oxy', 'omnicom group': 'omc', 'oneok': 'oke', 'oracle corp.': 'orcl', 'paccar inc.': 'pcar', 'packaging corporation of america': 'pkg', 'parker-hannifin': 'ph', 'paychex inc.': 'payx', 'paypal': 'pypl', 'pentair ltd.': 'pnr', "people's united financial": 'pbct', 'pepsico inc.': 'pep', 'perkinelmer': 'pki', 'perrigo': 'prgo', 'pfizer inc.': 'pfe', 'pg&e corp.': 'pcg', 'philip morris international': 'pm', 'phillips 66': 'psx', 'pinnacle west capital': 'pnw', 'pioneer natural resources': 'pxd', 'pnc financial services': 'pnc', 'polo ralph lauren corp.': 'rl', 'ppg industries': 'ppg', 'ppl corp.': 'ppl', 'praxair inc.': 'px', 'principal financial group': 'pfg', 'procter & gamble': 'pg', 'progressive corp.': 'pgr', 'prologis': 'pld', 'prudential financial': 'pru', 'public serv. enterprise inc.': 'peg', 'public storage': 'psa', 'pulte homes inc.': 'phm', 'pvh corp.': 'pvh', 'qorvo': 'qrvo', 'qualcomm inc.': 'qcom', 'quanta services inc.': 'pwr', 'quest diagnostics': 'dgx', 'range resources corp.': 'rrc', 'raymond james financial inc.': 'rjf', 'raytheon co.': 'rtn', 'realty income corporation': 'o', 'red hat inc.': 'rht', 'regency centers corporation': 'reg', 'regeneron': 'regn', 'regions financial corp.': 'rf', 'republic services inc': 'rsg', 'resmed': 'rmd', 'robert half international': 'rhi', 'rockwell automation inc.': 'rok', 'rockwell collins': 'col', 'roper technologies': 'rop', 'ross stores': 'rost', 'royal caribbean cruises ltd': 'rcl', 's&p global, inc.': 'spgi', 'salesforce.com': 'crm', 'sba communications': 'sbac', 'scana corp': 'scg', 'schlumberger ltd.': 'slb', 'seagate technology': 'stx', 'sealed air': 'see', 'sempra energy': 'sre', 'sherwin-williams': 'shw', 'simon property group inc': 'spg', 'skyworks solutions': 'swks', 'sl green realty': 'slg', 'snap-on inc.': 'sna', 'southern co.': 'so', 'southwest airlines': 'luv', 'stanley black & decker': 'swk', 'starbucks corp.': 'sbux', 'state street corp.': 'stt', 'stericycle inc': 'srcl', 'stryker corp.': 'syk', 'suntrust banks': 'sti', 'svb financial': 'sivb', 'symantec corp.': 'symc', 'synchrony financial': 'syf', 'synopsys inc.': 'snps', 'sysco corp.': 'syy', 't. rowe price group': 'trow', 'take-two interactive': 'ttwo', 'tapestry, inc.': 'tpr', 'target corp.': 'tgt', 'te connectivity ltd.': 'tel', 'technipfmc': 'fti', 'texas instruments': 'txn', 'textron inc.': 'txt', 'the bank of new york mellon corp.': 'bk', 'the clorox company': 'clx', 'the cooper companies': 'coo', 'the hershey company': 'hsy', 'the mosaic company': 'mos', 'the travelers companies inc.': 'trv', 'the walt disney company': 'dis', 'thermo fisher scientific': 'tmo', 'tiffany & co.': 'tif', 'time warner inc.': 'twx', 'tjx companies inc.': 'tjx', 'torchmark corp.': 'tmk', 'total system services': 'tss', 'tractor supply company': 'tsco', 'transdigm group': 'tdg', 'tripadvisor': 'trip', 'twenty-first century fox class a': 'foxa', 'twenty-first century fox class b': 'fox', 'tyson foods': 'tsn', 'u.s. bancorp': 'usb', 'udr inc': 'udr', 'ulta beauty': 'ulta', 'under armour class a': 'uaa', 'under armour class c': 'ua', 'union pacific': 'unp', 'united continental holdings': 'ual', 'united health group inc.': 'unh', 'united parcel service': 'ups', 'united rentals, inc.': 'uri', 'united technologies': 'utx', 'universal health services, inc.': 'uhs', 'unum group': 'unm', 'v.f. corp.': 'vfc', 'valero energy': 'vlo', 'varian medical systems': 'var', 'ventas inc': 'vtr', 'verisign inc.': 'vrsn', 'verisk analytics': 'vrsk', 'verizon communications': 'vz', 'vertex pharmaceuticals inc': 'vrtx', 'viacom inc.': 'viab', 'visa inc.': 'v', 'vornado realty trust': 'vno', 'vulcan materials': 'vmc', 'wal-mart stores': 'wmt', 'walgreens boots alliance': 'wba', 'waste management inc.': 'wm', 'waters corporation': 'wat', 'wec energy group inc': 'wec', 'wells fargo': 'wfc', 'western digital': 'wdc', 'western union co': 'wu', 'westrock company': 'wrk', 'weyerhaeuser corp.': 'wy', 'whirlpool corp.': 'whr', 'williams cos.': 'wmb', 'willis towers watson': 'wltw', 'wyndham worldwide': 'wyn', 'wynn resorts ltd': 'wynn', 'xcel energy inc': 'xel', 'xerox corp.': 'xrx', 'xilinx inc': 'xlnx', 'xl capital': 'xl', 'xylem inc.': 'xyl', 'yum! brands inc': 'yum', 'zimmer biomet holdings': 'zbh', 'zions bancorp': 'zion', 'zoetis': 'zts'}

fillerWords = ['more', 'any', 'your', 'youre', 'both', 'other', 'how', 'them', 'their', 'has', 'above', 'here', 'did', 'this', 'once', 'ours', 'which', 'again', 'further', 'there', 'its', 'where', 'yours', 'does', 'through', 'during', 'thatll', 'shouldve', 'him', 'but', 'hers', 'from', 'himself', 'didn', 'were', 'just', 'theirs', 'youll', 'about', 'was', 'below', 'such', 'that', 'under', 'the', 'yourself', 'after', 'before', 'own', 'his', 'can','are', 'then', 'all', 'those', 'yourselves', 'being', 'out', 'these', 'between', 'had', 'down', 'why', 'some', 'should', 'themselves', 'while', 'she', 'when', 'you', 'over', 'same', 'shes', 'ourselves', 'who', 'only', 'have', 'very', 'her', 'having', 'each', 'been', 'our', 'myself', 'youve', 'because', 'with', 'than', 'will', 'herself', 'itself', 'doing', 'too', 'they', 'what', 'few', 'off', 'wasn', 'isn', 'youd', 'whom', 'for', 'most', 'and', 'as', 'quite']

