import Button from '@mui/material/Button';
import Box from '@mui/material/Box';
import Tabs from '@mui/material/Tabs';
import Tab from '@mui/material/Tab';
import * as React from 'react';
import TextField from '@mui/material/TextField';
import Paper from '@mui/material/Paper';
import Typography from '@mui/material/Typography';
import Switch from '@mui/material/Switch';
import { DemoContainer } from '@mui/x-date-pickers/internals/demo';
import { AdapterDayjs } from '@mui/x-date-pickers/AdapterDayjs';
import { LocalizationProvider } from '@mui/x-date-pickers/LocalizationProvider';
import { DatePicker } from '@mui/x-date-pickers/DatePicker';
import dayjs from 'dayjs'
import axios from 'axios';


export default function SelecaoOpcoes({}) {
    const [tab, setTab] = React.useState(0)
    const [pSelected, setPSelected] = React.useState(true)
    const [riskFree, setRiskFree] = React.useState(0.10)
    const [periodos, setPeriodos] = React.useState(1)
    const [preco, setPreco] = React.useState(63)
    const [inicio, setInicio] = React.useState(dayjs('2022-04-17'));
    const [stock, setStock] = React.useState('VALE3.SA')
    const [loading, setLoading] = React.useState(false)

    const [resultado, setResultado] = React.useState('')

  const handleChange = (event, newValue) => {
      console.log(newValue)
    setTab(newValue);
  };

  React.useEffect(()=> {
      console.log(inicio)
      console.log(inicio.toISOString().substring(0, 10))
      },[inicio])

  function calcular_opcao() {
      setLoading(true)
      let tipo_opcao = '1' //europeia option
      if (tab == 0) {//asian option
        tipo_opcao = '2'
      }
      let url = "http://localhost:8000/calcular?cod_stock="
      url = url + stock
      url = url + "&preco_exercicio=" + preco.toString()
      url = url + "&tipo=" + tipo_opcao
      url = url + "&p_c="
      if (pSelected) {
        url = url + "p"
      }
    else {
        url = url + "c"
      }
      url = url + "&inicio=" + inicio.toISOString().substring(0, 10)

      url = url + "&risk_free=" + riskFree.toString().replace(',', '.')
      url = url + "&tempo=" + periodos.toString()
     console.log('url')
     console.log(url)


     axios.get(url).then((res) => {
         setResultado(res.data)

    console.log(res.data)
    }).catch((err) => {
        console.log(err)
        }).finally(()=> {
            setLoading(false)
            })



//      http://localhost:8000/?cod_stock=VALE3.SA&preco_exercicio=63&tipo=2&p_c=p&inicio=2022-04-17&risk_free=01
//      http://localhost:8000/?cod_stock=VALE3.SA&preco_exercicio=63&tipo=1&p_c=c&inicio=2020-02-02&risk_free=0.1065&tempo=1


  }

    return  <Paper sx={{
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                justifyContent: 'center'

            }}>
            <Box>
            <Tabs sx={{width: '100%'}} value={tab} onChange={handleChange}>
                <Tab label="Opções Asiaticas" />
                <Tab label="Opções Europeias" />
            </Tabs>
            </Box>

            <TextField label="Stock Name"
            value={stock}
                onChange={(evt) => {
                    setStock(evt.target.value)
                }}
            variant="filled" sx={{
                mt: 2,
                mb: 1
                }} />

            <TextField
            label="Preco em Exercicio"
            value={preco}
                onChange={(evt) => {
                    setPreco(evt.target.value)
                }}
            variant="filled" sx={{
                mb: 1
                }} />

            <TextField
            label="Risk Free"
            value={riskFree}
                onChange={(evt) => {
                    setRiskFree(evt.target.value)
                }}
            type="number" variant="filled" sx={{
                mb: 1
                }} />

            <TextField
                label="Periodos"
                value={periodos}
                onChange={(evt) => {
                    setPeriodos(evt.target.value)
                }}
                type="number" variant="filled" sx={{
                mb: 1
                }} />


            <LocalizationProvider dateAdapter={AdapterDayjs}>
              <DemoContainer components={['DatePicker', 'DatePicker']}>

                <DatePicker
                  label="Inicio (MES/DIA/ANO)"
                  value={inicio}
                  onChange={(newValue) => setInicio(newValue)}
                />
              </DemoContainer>
            </LocalizationProvider>


            <Box sx={{display: 'flex', flexDirection: 'row', mt: 1, mb: 1}}>
                <Paper sx={{padding: 1, background: pSelected ? '' : 'green'}} elevation={pSelected == 0 ? 8 : 1}>
                <Typography sx={{mt: 0}}>C</Typography>
                </Paper>
                <Switch checked={pSelected} onChange={()=>{
                    setPSelected(!pSelected)
                }}/>

                <Paper sx={{padding: 1, background: pSelected ? 'green' : ''}} elevation={pSelected == 0 ? 1 : 8}>
                <Typography sx={{mt: 0}}>P</Typography>
                </Paper>

            </Box>


            {
                tab == 0 ?
                <Box>

                </Box>

                :
                <Box>

                </Box>
            }

        <Paper
            elevation={9}
        sx={{
            mt: 1,
            width: '100%'
            }}>
            <Typography>
                Resultado: {loading ? '...' : resultado}
            </Typography>
        </Paper>



            <Button
            disabled={loading}
            onClick={()=>{
                calcular_opcao()
            }}
            sx={{width: '100%', mt: 2}} variant='contained'>
                Consultar
            </Button>
        </Paper>
    }